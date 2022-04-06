import random, pickle, shutil, json, xlrd, scipy.io as sciio, numpy as np, os
from PIL import Image
from tqdm import trange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

import comparison0, label0, utility0


class DatasetFromPath(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __getitem__(self, index):
        image, label, label_c = self.image_path[index]

        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_path)


def AllDatasets(switch_dataset, path_all_dataset, path_all_dataset_usb, switch_imbalance):
    def LeavesPath(dir_root, shuffle=False):  # root/height1/height2_/height3_/.../image
        #  height1     0               0
        #             / \             / \
        #  height2   0   1    --->   1   2
        #               / \             / \
        #  height3     0   1           3   4
        #             /               /
        #           image           image
        #
        #           leaves_array:
        #           height=3;       leaves:1,3,4;
        #                           node(节点):0~4;

        def f1(dir_height1):
            for i_height2 in os.listdir(dir_height1):
                dir_height2 = dir_height1 + f'{i_height2}/'
                if os.path.isdir(dir_height2):
                    f1(dir_height2)
                else:
                    leaves_path.append(dir_height1)
                    break

        leaves_path = []  # 索引：叶子（细类）；值：叶子路径
        dir_height1 = dir_root + 'height1/'
        f1(dir_height1)

        if shuffle:  # shuffle aims to seperate fine classes that belong to a same coarse class
            random.shuffle(leaves_path)
        return leaves_path

    def LeavesArray(leaves_path):  # 横向索引：叶子，对应leaves_path；纵向索引：height - 1；值：标签，值为-1表示叶子下的图片，值为-2表示无
        max_height = 1
        for i in range(len(leaves_path)):
            segment = 0
            for j in leaves_path[i].split('height1/')[1].split('/'):
                if len(j) > 0:
                    segment += 1
            height = segment + 1
            if height > max_height:
                max_height = height

        leaves_array = np.full((len(leaves_path), 1), 0, 'int')

        for i_height in range(max_height):
            layer_node = []
            for i_leaf in range(len(leaves_path)):
                str_a = leaves_path[i_leaf].split('height1/')[1]
                list_a = str_a.split('/')

                try:
                    if len(list_a[i_height]) > 0:
                        layer_node.append(list_a[i_height])
                except:
                    print(end='')

                # layer_node = list(set(layer_node))  # 排序会乱
                new = []
                for i in range(len(layer_node)):
                    if layer_node[i] not in new:
                        new.append(layer_node[i])
                layer_node = new
            # print('layer_node', layer_node, len(layer_node))

            layer_labels = []
            for i_leaf in range(len(leaves_path)):
                str_a = leaves_path[i_leaf].split('height1/')[1]
                list_a = str_a.split('/')

                try:
                    len1 = len(list_a[i_height])
                    if len1 == 0:
                        layer_labels.append(-1)
                        continue
                except:
                    layer_labels.append(-2)
                    continue

                for i in range(len(layer_node)):
                    if list_a[i_height] == layer_node[i]:
                        layer_labels.append(i)

            # print('layer_labels', layer_labels, len(layer_labels))
            leaves_array = np.hstack((leaves_array, np.array(layer_labels).reshape((-1, 1))))
            # print(leaves_array)
        # np.set_printoptions(threshold=np.inf)
        # print(leaves_array)
        return leaves_array

    def TreeFolderResize(flag_train, size_a, path_dataset):
        dir_root = path_dataset + flag_train  # root/height1/height2_/height3_/.../image
        flag_train = flag_train.split('/')[0]
        leaves_path = LeavesPath(dir_root)
        for i_fine in trange(len(leaves_path)):
            dir_fine = leaves_path[i_fine]
            new_dir_fine = dir_fine.replace(f'{flag_train}/height1/', f'resize{size_a}/{flag_train}/height1/')
            os.makedirs(new_dir_fine) if not os.path.exists(new_dir_fine) else ''

            list_image = []
            for i in os.listdir(dir_fine):
                list_image.append(i)
            for i_image in range(len(list_image)):
                dir_image = dir_fine + list_image[i_image]
                new_dir_image = new_dir_fine + list_image[i_image]

                image = Image.open(dir_image)
                image = image.convert('RGB')
                if flag_train == 'test':
                    transform1 = transforms.Compose([
                        transforms.Resize(size_a),
                        transforms.CenterCrop(round(size_a * 7 / 8)),  # 测试样本的大小与训练样本一样，且不需要数据增广
                    ])
                else:
                    transform1 = transforms.Compose([
                        transforms.Resize(size_a),  # 宽高等比例缩放：将较短的边设置为size_a，然后较长的边等比例缩放
                        # transforms.Resize((size_a, size_a)),  # 宽高非等比例缩放：将较短和较长的边都设置为size_a
                    ])
                transform1(image).save(new_dir_image)

    def TreeSum(p):  # p: .../height/
        s = 0
        for i in os.listdir(p):
            coarse = 0
            for j in os.listdir(f'{p}{i}/'):
                fine = 0
                for k in os.listdir(f'{p}{i}/{j}/'):
                    fine += 1
                print('fine', fine)
                coarse += fine
            print('coarse', coarse)
            s += coarse
        print('sum', s)

    switch_augument = ['Augument0', 'Augument1', 'Augument2', 'Augument3'][0]

    if 'CIFAR' in switch_dataset:
        flag_a = False
        if flag_a:
            def CIFARImageFolder(dataset, train, isCoarse=True, isChinese=True, long_tail=False):
                # dataset: 'CIFAR-10'; 'CIFAR-100'.  train: 'train'; 'test'
                if dataset == 'CIFAR-10':
                    new_path_dataset = path_all_dataset + 'CIFAR-10_tree'

                    if train == 'train':
                        dataset = datasets.CIFAR10(root=path_all_dataset, train=True, download=False)

                    data = dataset.data
                    labels = dataset.targets
                    name_f_classes = dataset.classes

                    print(name_f_classes)

                elif dataset == 'CIFAR-100':
                    path_dataset = path_all_dataset + 'cifar-100-python'
                    new_path_dataset = path_all_dataset + 'CIFAR-100_tree'

                    relation_f = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6,
                                  13, 15, 3, 15, 0, 11,
                                  1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0,
                                  17, 4, 18, 17, 10, 3, 2, 12, 12,
                                  16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8,
                                  19, 18, 1, 2, 15, 6, 0, 17, 8, 14,
                                  13
                                  ]

                    name_c_classes_ = \
                        ['水生哺乳动物', '鱼', '花卉', '食品容器', '水果和蔬菜', '家用电器', '家庭家具', '昆虫', '大型食肉动物',
                         '大型人造户外用品', '大自然户外场景', '大型杂食动物和食草动物', '中型哺乳动物', '非昆虫无脊椎动物',
                         '人', '爬行动物', '小型哺乳动物', '树木', '车辆一', '车辆二']
                    name_f_classes_ = \
                        ['苹果', '水族馆鱼', '宝贝', '熊', '海狸', '床', '蜜蜂', '甲虫', '自行车', '瓶子', '碗', '男孩', '桥',
                         '公共汽车', '蝴蝶', '骆驼', '罐', '城堡', '毛毛虫', '牛', '椅子', '黑猩猩', '时钟', '云', '蟑螂', '沙发',
                         '螃蟹', '鳄鱼', '杯子', '恐龙', '海豚', '大象', '比目鱼', '森林', '狐狸', '女孩', '仓鼠', '房子', '袋鼠',
                         '键盘', '台灯', '割草机', '豹', '狮子', '蜥蜴', '龙虾', '男人', '枫树', '摩托车', '山', '老鼠', '蘑菇',
                         '橡树', '橘子', '兰花', '水獭', '棕榈树', '梨', '皮卡车', '松树', '平原', '盘子', '罂粟花', '豪猪',
                         '负鼠', '兔子', '浣熊', '鳐', '路', '火箭', '玫瑰', '海', '海豹', '鲨鱼', '地鼠', '臭鼬',
                         '摩天大楼', '蜗牛', '蛇', '蜘蛛', '松鼠', '有轨电车', '向日葵', '甜辣椒', '桌子', '坦克', '电话', '电视机',
                         '老虎', '拖拉机', '火车', '鳟鱼', '郁金香', '乌龟', '衣柜', '鲸鱼', '柳树', '狼', '女人', '蠕虫']

                with open(path_dataset + '/meta', 'rb') as fo:
                    dict_meta = pickle.load(fo, encoding='latin1')
                with open(path_dataset + f'/{train}', 'rb') as fo:  # train; test
                    dict_train = pickle.load(fo, encoding='latin1')

                # for i in dict_meta:
                #     print(i)
                #     print(type(dict_meta[i]), len(dict_meta[i]))
                #     # fine_label_names;   coarse_label_names
                #     # <class 'list'> 100; <class 'list'> 20

                # for i in dict_train:
                #     print(i)
                #     print(type(dict_train[i]), len(dict_train[i]))
                #     # filenames;            batch_label;      fine_labels;          coarse_labels;        data
                #     # <class 'list'> 50000; <class 'str'> 21; <class 'list'> 50000; <class 'list'> 50000; <class 'numpy.ndarray'> 50000

                shutil.rmtree(new_path_dataset) if os.path.exists(new_path_dataset) else ''
                os.makedirs(new_path_dataset) if not os.path.exists(new_path_dataset) else ''

                name_f_classes = dict_meta['fine_label_names']
                name_c_classes = dict_meta['coarse_label_names']

                if not isChinese:
                    name_c_classes_ = ['' * len(name_c_classes)]
                    name_f_classes_ = ['' * len(name_f_classes)]

                for i_f in range(len(name_f_classes)):
                    i_c = relation_f[i_f]
                    i_c_name = name_c_classes[i_c]
                    i_c_name_ = name_c_classes_[i_c]
                    i_f_name = name_f_classes[i_f]
                    i_f_name_ = name_f_classes_[i_f]

                    if isCoarse:
                        path = new_path_dataset + f'hright1/height2_c{i_c}_{i_c_name}_{i_c_name_}/' + \
                               f'f{i_f}_{i_f_name}_{i_f_name_}/'
                    else:
                        path = new_path_dataset + f'f{i_f}_{i_f_name}_{i_f_name_}/'
                    os.makedirs(path) if not os.path.exists(path) else ''

                data = dict_train['data']  # (50000, 3072)
                labels = dict_train['fine_labels']  # 50000
                if long_tail:
                    count = np.zeros(len(name_f_classes), dtype=np.int64)
                    new_labels = []
                    for i in trange(len(labels)):
                        if count[labels[i]] < long_tail[labels[i]]:
                            count[labels[i]] += 1

                            if i == 0:
                                new_data = data[i].reshape(1, -1)
                            else:
                                new_data = np.concatenate((new_data, data[i].reshape(1, -1)))
                            new_labels.append(labels[i])
                        else:
                            continue

                    data = new_data
                    labels = new_labels

                for i in trange(data.shape[0]):
                    img = np.reshape(data[i], (3, 32, 32))  # pickle
                    i0 = Image.fromarray(img[0])
                    i1 = Image.fromarray(img[1])
                    i2 = Image.fromarray(img[2])
                    img = Image.merge('RGB', (i0, i1, i2))

                    i_f = labels[i]

                    i_c = relation_f[i_f]
                    i_c_name = name_c_classes[i_c]
                    i_c_name_ = name_c_classes_[i_c]
                    i_f_name = name_f_classes[i_f]
                    i_f_name_ = name_f_classes_[i_f]

                    if isCoarse:
                        path = new_path_dataset + f'height1/height2_c{i_c}_{i_c_name}_{i_c_name_}/' + \
                               f'f{i_f}_{i_f_name}_{i_f_name_}/'
                    else:
                        path = new_path_dataset + f'f{i_f}_{i_f_name}_{i_f_name_}/'

                    img.save(path + dict_train['filenames'][i])

            CIFARImageFolder(switch_dataset, 'train')
            CIFARImageFolder(switch_dataset, 'test')

        # nine cows
        flag_nine_cows = False
        if flag_nine_cows:
            # torchvision vs pickle

            # torchvision
            train_dataset = datasets.CIFAR100(root=path_all_dataset, train=True, download=False)
            # print(train_dataset.data.shape)  # (50000, 32, 32, 3)

            a = train_dataset.data[0]
            # print(a.shape)  # (32, 32, 3)
            # utility0.ImageFromDataNumpy(a)  # a cow

            # a1 = a.reshape(3, 32, 32)
            # print(a1.shape)  # (3, 32, 32)
            # utility0.ImageFromDataNumpy(a1)  # mosaic

            # a2 = a.transpose((2, 0, 1))
            # print(a2.shape)  # (3, 32, 32)
            # utility0.ImageFromDataNumpy(a2)  # a cow

            # pickle
            name_dataset = 'cifar-100-python'
            path_dataset = path_all_dataset + name_dataset
            with open(path_dataset + '/train', 'rb') as fo:
                dict_train = pickle.load(fo, encoding='latin1')
            train_data = dict_train['data']
            # print(train_data.shape)  # (50000, 3072)

            a = train_data[0]
            # print(a.shape)  # (3072,)

            a1 = a.reshape(32, 32, 3)
            # print(a1.shape)  # (32, 32, 3)
            # utility0.ImageFromDataNumpy(a1)  # nine cows

            a2 = a.reshape(3, 32, 32)
            # print(a2.shape)  # (3, 32, 32)
            # utility0.ImageFromDataNumpy(a2)  # a cow

        path_dataset = path_all_dataset + f'{switch_dataset}_tree/'

        # =========================================================
        train_a, test_a = 'train/', 'test/'
        leaves_path = LeavesPath(path_dataset + train_a, shuffle=True)
        # 对于一个数据集，无论有几层，预先选定某一层作为训练时的粗类。对于一个细类，评价时有可以有多层粗类，但训练时只有一层粗类
        height = 2  # 指定某个height的粗类为训练时的粗类

        leaves_array = LeavesArray(leaves_path)
        coarse_layer = leaves_array[:, height - 1: height].flatten().tolist()
        image_path = [[os.path.join(leaves_path[i], j), i, coarse_layer[i]]
                      for i in range(len(leaves_path)) for j in os.listdir(leaves_path[i])]  # [image fine coarse]

        label = [i[1] for i in image_path]
        label_c = [i[2] for i in image_path]
        num_cls, num_cls_c, relation = label0.RelationFine1(label, label_c)
        eval_tree = utility0.ArrayToTree(leaves_array)

        leaves_path_test = [i.replace(train_a, test_a) for i in leaves_path]
        image_path_test = [[os.path.join(leaves_path_test[i], j), i, coarse_layer[i]]
                           for i in range(len(leaves_path_test)) for j in os.listdir(leaves_path_test[i])]

        if switch_augument == 'Augument0':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32
                transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化的均值和方差
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            raise ValueError(switch_augument)

    if 'VOC2012' in switch_dataset:
        path_dataset = path_all_dataset + 'VOC2012_tree/'

        size_a = False  # False;36;64;128;256
        if size_a:
            TreeFolderResize('train/', size_a, path_dataset)
            TreeFolderResize('val/', size_a, path_dataset)
            e

        # =========================================================
        size_a = 36

        train_a, test_a = f'resize{size_a}/train/', f'resize{size_a}/val/'
        leaves_path = LeavesPath(path_dataset + train_a, shuffle=True)
        if switch_dataset.split('VOC2012')[1] == '_Per':
            leaves_path = [i for i in leaves_path if 'person' not in i]
            # 对于一个数据集，无论有几层，预先选定某一层作为训练时的粗类。对于一个细类，评价时有可以有多层粗类，但训练时只有一层粗类
            height = 2  # 指定某个height的粗类为训练时的粗类
        elif switch_dataset.split('VOC2012')[1] == '_PerBir':
            leaves_path = [i for i in leaves_path if 'person' not in i and 'bird' not in i]
            # 对于一个数据集，无论有几层，预先选定某一层作为训练时的粗类。对于一个细类，评价时有可以有多层粗类，但训练时只有一层粗类
            height = 3  # 指定某个height的粗类为训练时的粗类

        leaves_array = LeavesArray(leaves_path)
        coarse_layer = leaves_array[:, height - 1: height].flatten().tolist()
        image_path = [[os.path.join(leaves_path[i], j), i, coarse_layer[i]]
                      for i in range(len(leaves_path)) for j in os.listdir(leaves_path[i])]  # [image fine coarse]

        # the original dataset that is long-tailed whose label0 isn't head, we need label0 is head
        label = [i[1] for i in image_path]
        a, b = label0.Sort_HeadTail(label, switch_imbalance)
        label = [i - a[np.where(b == i)[0][0]] for i in label]
        for i in range(len(image_path)):
            image_path[i][1] = label[i]

        label_c = [i[2] for i in image_path]
        num_cls, num_cls_c, relation = label0.RelationFine1(label, label_c)
        eval_tree = utility0.ArrayToTree(leaves_array)

        leaves_path_test = [i.replace(train_a, test_a) for i in leaves_path]
        image_path_test = [[os.path.join(leaves_path_test[i], j), i, coarse_layer[i]]
                           for i in range(len(leaves_path_test)) for j in os.listdir(leaves_path_test[i])]
        label = [i[1] for i in image_path_test]
        label = [i - a[np.where(b == i)[0][0]] for i in label]
        for i in range(len(image_path_test)):
            image_path_test[i][1] = label[i]

        if switch_augument == 'Augument0':
            transform = transforms.Compose([
                transforms.RandomCrop(round(size_a * 7 / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            transform_test = transforms.Compose([
                transforms.CenterCrop(round(size_a * 7 / 8)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            raise ValueError(switch_augument)

    if switch_dataset == 'SUN397':
        path_dataset = path_all_dataset + 'SUN397_tree/'

        flag_a = False  # SUN397/ to SUN397_tree/
        if flag_a:
            path_dataset = path_all_dataset + 'SUN397/'
            book = xlrd.open_workbook(path_dataset + 'three_levels.xlsx')
            sheet1 = book.sheets()[0]

            xlsx_f, xlsx_c1, xlsx_c2, xlsx_path = [], [], [], []
            for i in range(sheet1.nrows):
                if i >= 2:
                    row1 = sheet1.row_values(i)  # 输出第i行的所有值
                    path = row1[0].split('/', 1)[1] + '/'
                    coarse_level1 = row1[1: 4]
                    coarse_level2 = row1[4:]
                    index1 = np.where(np.array(coarse_level1) == 1.0)[0]
                    index2 = np.where(np.array(coarse_level2) == 1.0)[0]
                    if sum(coarse_level2) == 1.0:  # single label depending on level2
                        xlsx_f.append(int(i - 2))
                        xlsx_c1.append(index1[0])
                        xlsx_c2.append(index2[0])
                        xlsx_path.append(path)
                    else:
                        xlsx_f.append(-1)
                        xlsx_c1.append(-1)
                        xlsx_c2.append(-1)
                        xlsx_path.append(-1)

            for i, j, k, p in zip(xlsx_f, xlsx_c1, xlsx_c2, xlsx_path):
                print(i, j, k, p)
                # 0 2 11 a/abbey/
                # 1 0 3 a/airplane_cabin/
                # ...
                # 394 0 4 w/wrestling_ring/indoor/
                # -1 -1 -1 -1
                # 396 0 2 /y/youth_hostel/

            new_path_data = path_all_dataset + f'SUN397_tree/'
            shutil.rmtree(new_path_data) if os.path.exists(new_path_data) else ''
            for i_xlsx in trange(len(xlsx_f)):
                if xlsx_f[i_xlsx] != -1:
                    dir_fine = path_dataset + xlsx_path[i_xlsx]
                    new_dir_coarse = new_path_data + f'root/height1/height2_{xlsx_c1[i_xlsx]}/' + \
                                     f'height3_{xlsx_c2[i_xlsx]}/'
                    a = xlsx_path[i_xlsx].split('/')
                    if len(a) == 3:  # w/wave/
                        new_dir_fine = new_dir_coarse + a[1] + '/'
                        shutil.copytree(dir_fine, new_dir_fine)
                    elif len(a) == 4:  # w/waterfall/block/
                        new_dir_fine = new_dir_coarse + f'{a[1]}_{a[2]}/'
                        shutil.copytree(dir_fine, new_dir_fine)

        size_a = False  # False;36;64;128;256
        if size_a:
            TreeFolderResize('root/', size_a, path_dataset)

        # =========================================================
        size_a = 36
        switch_imbalance = 'Original'

        train_a = f'resize{size_a}/root/'
        leaves_path = LeavesPath(path_dataset + train_a, shuffle=True)
        # 对于一个数据集，无论有几层，预先选定某一层作为训练时的粗类。对于一个细类，评价时有可以有多层粗类，但训练时只有一层粗类
        height = 3  # 指定某个height的粗类为训练时的粗类

        leaves_array = LeavesArray(leaves_path)
        coarse_layer = leaves_array[:, height - 1: height].flatten().tolist()
        image_path = [[os.path.join(leaves_path[i], j), i, coarse_layer[i]]
                      for i in range(len(leaves_path)) for j in os.listdir(leaves_path[i])]

        # the original dataset that is long-tailed whose label0 isn't head, we need label0 is head
        label = [i[1] for i in image_path]
        a, b = label0.Sort_HeadTail(label, switch_imbalance)
        label = [i - a[np.where(b == i)[0][0]] for i in label]
        for i in range(len(image_path)):
            image_path[i][1] = label[i]

        label = [i[1] for i in image_path]
        label_c = [i[2] for i in image_path]
        num_cls, num_cls_c, relation = label0.RelationFine1(label, label_c)
        eval_tree = utility0.ArrayToTree(leaves_array)

        image_path_test = 'None'

        if switch_augument == 'Augument0':
            transform = transforms.Compose([
                transforms.RandomCrop(round(size_a * 7 / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            transform_test = transforms.Compose([
                transforms.CenterCrop(round(size_a * 7 / 8)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            raise ValueError(switch_augument)

    if switch_dataset == 'iNaturalist2017':
        # p = 'iNaturalist2017/train_val_images/'
        # p = 'iNaturalist2017_tree/resize36/train/height1/'
        # p = 'iNaturalist2017_tree/resize36/val/height1/'
        # TreeSum(path_all_dataset + p)

        size_a = False  # False;36;64;128;256
        if size_a:  # iNaturalist
            def SplitTrainVal_json(size_a, path_data, new_path_data):  # json iNaturalist2017
                # val.json is used but there are sone errors on train.json
                # train_val_images: 675170 images, val,json: 95986 images; train.json: 166291
                # 675170 - 95986 = 579184
                with open(path_data + 'val2017.json', 'r', encoding='utf-8') as f:
                    json_a = json.load(f)

                b = list(json_a.values())
                c = b[1]
                l_val = []
                for i in range(len(c)):
                    a = c[i].get('file_name')  # 'train_val_images/Mammalia/Marmota flaviventris/3c919dfdcfe7837b420d3dc640c21c70.jpg'
                    l_val.append(a.split('/', 1)[1])
                # print(l_val, len(l_val))  # ['Mammalia/Marmota flaviventris/3c919dfdcfe7837b420d3dc640c21c70.jpg', ...] 95986

                old_dir = path_all_dataset +  'iNaturalist2017/train_val_images/'
                l_all, l_train = [], []
                for i in os.listdir(old_dir):  # i: coarse
                    for j in os.listdir(old_dir + f'{i}/'):  # j: fine
                        for k in os.listdir(old_dir + f'{i}/{j}/'):  # k: image
                            l_all.append(f'{i}/{j}/{k}')
                # print(l_all, len(l_all))  # ['Mammalia/Marmota flaviventris/3c919dfdcfe7837b420d3dc640c21c70.jpg', ...] 675170

                new_dir_val = new_path_data + 'val/height1/'
                shutil.rmtree(new_dir_val) if os.path.exists(new_dir_val) else ''
                new_dir_train = new_path_data + 'train/height1/'
                shutil.rmtree(new_dir_train) if os.path.exists(new_dir_train) else ''

                for i in trange(len(l_all)):
                    image = Image.open(old_dir + l_all[i])
                    image = image.convert('RGB')


                    if l_all[i] in l_val:
                        transform1 = transforms.Compose([
                            transforms.Resize(size_a),
                            transforms.CenterCrop(round(size_a * 7 / 8)),  # 测试样本的大小与训练样本一样，且不需要数据增广
                        ])

                        a = (new_dir_val + l_all[i]).rsplit('/', 1)[0]
                        os.makedirs(a) if not os.path.exists(a) else ''
                        transform1(image).save(new_dir_val + l_all[i])
                    else:
                        transform1 = transforms.Compose([
                            transforms.Resize(size_a),  # 宽高等比例缩放：将较短的边设置为size_a，然后较长的边等比例缩放
                            # transforms.Resize((size_a, size_a)),  # 宽高非等比例缩放：将较短和较长的边都设置为size_a
                        ])

                        a = (new_dir_train + l_all[i]).rsplit('/', 1)[0]
                        os.makedirs(a) if not os.path.exists(a) else ''
                        transform1(image).save(new_dir_train + l_all[i])

            path_data = path_all_dataset + 'iNaturalist2017/'
            new_path_data = path_all_dataset + f'iNaturalist2017_tree/resize{size_a}/'
            SplitTrainVal_json(size_a, path_data, new_path_data)

        path_dataset = path_all_dataset + 'iNaturalist2017_tree/'

        # =========================================================
        size_a = 36
        switch_imbalance = 'Original'

        train_a, test_a = f'resize{size_a}/train/', f'resize{size_a}/val/'
        leaves_path = LeavesPath(path_dataset + train_a, shuffle=True)
        # 对于一个数据集，无论有几层，预先选定某一层作为训练时的粗类。对于一个细类，评价时有可以有多层粗类，但训练时只有一层粗类
        height = 2  # 指定某个height的粗类为训练时的粗类

        leaves_array = LeavesArray(leaves_path)
        coarse_layer = leaves_array[:, height - 1: height].flatten().tolist()
        image_path = [[os.path.join(leaves_path[i], j), i, coarse_layer[i]]
                      for i in range(len(leaves_path)) for j in os.listdir(leaves_path[i])]

        # the original dataset that is long-tailed whose label0 isn't head, we need label0 is head
        label = [i[1] for i in image_path]
        a, b = label0.Sort_HeadTail(label, switch_imbalance)
        label = [i - a[np.where(b == i)[0][0]] for i in label]
        for i in range(len(image_path)):
            image_path[i][1] = label[i]

        label_c = [i[2] for i in image_path]
        num_cls, num_cls_c, relation = label0.RelationFine1(label, label_c)
        eval_tree = utility0.ArrayToTree(leaves_array)

        leaves_path_test = [i.replace(train_a, test_a) for i in leaves_path]
        image_path_test = [[os.path.join(leaves_path_test[i], j), i, coarse_layer[i]]
                           for i in range(len(leaves_path_test)) for j in os.listdir(leaves_path_test[i])]
        label = [i[1] for i in image_path_test]
        label = [i - a[np.where(b == i)[0][0]] for i in label]
        for i in range(len(image_path_test)):
            image_path_test[i][1] = label[i]

        if switch_augument == 'Augument0':
            transform = transforms.Compose([
                transforms.RandomCrop(round(size_a * 7 / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            transform_test = transforms.Compose([
                transforms.CenterCrop(round(size_a * 7 / 8)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            raise ValueError(switch_augument)

    if switch_dataset == 'tieredImageNet':
        path_dataset = path_all_dataset + 'tieredImageNet_tree/'

        # TreeSum(path_all_dataset + 'tieredImageNet_tree/resize36/root/height1/')

        flag_a = False  # generate root from a USB flash disk
        if flag_a:
            def FuseTrainVal(l0, path_dataset, new_dir):
                i_new_f, i_new_c = -1, -1
                for i_l0 in l0:
                    dir = path_dataset + i_l0

                    l_c = []
                    for i in os.listdir(dir):
                        l_c.append(int(i))
                    l_c.sort()

                    for i_c in trange(len(l_c)):
                        dir_coarse = dir + f'{l_c[i_c]}/'
                        i_new_c += 1
                        new_dir_coarse = new_dir + f'c{i_new_c}/'

                        l_f = []
                        for i in os.listdir(dir_coarse):
                            l_f.append(int(i))
                        l_f.sort()

                        for i_f in range(len(l_f)):
                            dir_fine = dir_coarse + f'{l_f[i_f]}/'
                            i_new_f += 1
                            new_dir_fine = new_dir_coarse + f'f{i_new_f}/'
                            os.makedirs(new_dir_fine) if not os.path.exists(new_dir_fine) else ''

                            l_data = []
                            for i in os.listdir(dir_fine):
                                l_data.append(i)
                            l_data.sort()

                            for i in range(len(l_data)):
                                image = Image.open(dir_fine + f'{l_data[i]}')
                                image = image.convert('RGB')
                                image.save(new_dir_fine + f'{i}.jpg')

            new_dir = path_dataset + 'root/height1/'
            shutil.rmtree(new_dir) if os.path.exists(new_dir) else ''
            os.makedirs(new_dir) if not os.path.exists(new_dir) else ''
            FuseTrainVal(['train/', 'test/', 'val/'], '/media/lzy/大U盘/数据集/tiered-imagenet/', new_dir)

        size_a = False  # False;36;64;84
        if size_a:
            TreeFolderResize('root/', size_a, path_dataset)

        # =========================================================
        size_a = 36

        train_a = f'resize{size_a}/root/'
        leaves_path = LeavesPath(path_dataset + train_a, shuffle=True)
        # 对于一个数据集，无论有几层，预先选定某一层作为训练时的粗类。对于一个细类，评价时有可以有多层粗类，但训练时只有一层粗类
        height = 2  # 指定某个height的粗类为训练时的粗类

        leaves_array = LeavesArray(leaves_path)
        coarse_layer = leaves_array[:, height - 1: height].flatten().tolist()
        image_path = [[os.path.join(leaves_path[i], j), i, coarse_layer[i]]
                      for i in range(len(leaves_path)) for j in os.listdir(leaves_path[i])]

        label = [i[1] for i in image_path]
        label_c = [i[2] for i in image_path]
        num_cls, num_cls_c, relation = label0.RelationFine1(label, label_c)
        eval_tree = utility0.ArrayToTree(leaves_array)

        image_path_test = 'None'

        if switch_augument == 'Augument0':
            if size_a == 84:
                transform = transforms.Compose([
                    transforms.RandomCrop(size_a, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                transform_test = transforms.Compose([
                    transforms.CenterCrop(size_a),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            else:
                transform = transforms.Compose([
                    transforms.RandomCrop(round(size_a * 7 / 8)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                transform_test = transforms.Compose([
                    transforms.CenterCrop(round(size_a * 7 / 8)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            raise ValueError(switch_augument)

    if switch_dataset == 'ImageNet2012':
        flag_a = False
        if flag_a:
            path_dataset = path_all_dataset + 'ImageNet2012/'
            book = xlrd.open_workbook(path_dataset + 'class_name.xls')
            sheet1 = book.sheets()[0]

            list_name = []
            for i in range(sheet1.nrows):
                row1 = sheet1.row_values(i)
                fine_name = row1[1]
                list_name.append(fine_name)
            # print(list_name)
            a, b = np.unique(np.array(list_name), return_counts=True)

            min_f = 3  # 3, 4
            index = np.where(np.array(b) >= min_f)[0]
            list_coarse = []
            for i in index:
                list_coarse.append(a[i])

            new_path_data = path_all_dataset + f'ImageNet2012_c_{min_f}f/'
            new_dir_root = new_path_data + f'/root/'
            shutil.rmtree(new_dir_root) if os.path.exists(new_dir_root) else ''

            for i_c in trange(len(list_coarse)):
                new_dir_coarse = new_dir_root + list_coarse[i_c] + '/'
                for i in range(sheet1.nrows):
                    row1 = sheet1.row_values(i)
                    name1, name_coarse = row1[0], row1[1]
                    if name_coarse == list_coarse[i_c]:
                        new_dir_fine = new_dir_coarse + name1 +'/'
                        shutil.copytree(path_dataset + f'ILSVRC2012_img_train/{name1}/', new_dir_fine)

        path_dataset_usb = path_all_dataset_usb + 'ImageNet2012/'
        path_dataset = path_all_dataset + 'ImageNet2012_tree/root/'

        train_dataset_mat = sciio.loadmat(path_dataset_usb  + 'tree.mat')
        tree = train_dataset_mat['tree']

        height = []
        for i in range(tree.shape[0]):
            height.append(tree[i, 1])
        a, b = utility0.Unique1(height, False)
        height_max = max(a) - min(a) + 1

        fine = []
        for i in range(tree.shape[0]):
            fine.append(tree[i, 1])
        a, b = utility0.Unique1(fine, False)
        fine_number = b[0]

        all_parent = []
        for i in range(tree.shape[0]):
            if i < fine_number:
                p, h = [], []

                p.append(i)
                h.append(tree[i, 1])
                parent = tree[i, 0]
                for j in range(height_max):
                    if parent == 0:
                        break
                    p.append(parent)
                    h.append(tree[parent - 1, 1])
                    parent = tree[parent - 1, 0]

                # print(p)
                # print(h)
                h0 = [i for i in range(height_max) if i not in h]
                # print(h0)
                for i in range(height_max):
                    if i in h0:
                        p.insert(i, -1)
                # print(p, len(p))

                all_parent.append(p)
        # print(all_parent)

        all_parent_array = np.full((len(all_parent), height_max), -2)
        n, m = 0, 0
        for i_f in range(len(all_parent)):
            for i_level in range(height_max - 1, -1, -1):
                all_parent_array[n, m] = all_parent[i_f][i_level]
                m += 1
            m = 0
            n += 1
        print(all_parent_array)

        def f1(l):
            a, b = utility0.Unique1(l, False)
            print(a, b)

        for i in range(all_parent_array.shape[1]):
            l = all_parent_array[:, i: i + 1].flatten().tolist()
            # if -1 not in l:
            #     print('height', i + 1)

            # f1(l)

            n_j = 0
            for j in l:
                if j == -1:
                    n_j += 1
            print('height', i + 1, '-1:', n_j)



        e
        name = []
        for i in os.listdir(path_dataset_usb + 'ILSVRC2012_img_train/'):
            name.append(i)
        name.sort()

        # shutil.rmtree(path_dataset) if os.path.exists(path_dataset) else ''
        all_path = []
        for i_f in range(len(all_parent)):
            path, height = path_dataset, 1
            for i_level in range(height_max - 1, -1, -1):
                t = all_parent[i_f][i_level]
                # print(t)
                if height == height_max:
                    path += f'height{height}_f{t}_{name[i_f]}/'
                else:
                    if t != -1:
                        path += f'height{height}_c{t}/'
                    else:
                        path += f'height{height}/'
                height += 1
            # print(path)
            # os.makedirs(path) if not os.path.exists(path) else ''
            all_path.append(path)
        print(all_path)

    if switch_dataset == 'DD':
        path_data = path_dataset + 'DD/DDTrain'
        train_dataset_mat = sciio.loadmat(path_data)
        train_dataset = train_dataset_mat['data_array']  # (3020, 474)
        train_data = train_dataset[:, :-1]
        train_labels = [int(i) - 1 for i in train_dataset[:, -1:].flatten().tolist()]

        path_data = path_dataset + 'DD/DDTest'
        test_dataset_mat = sciio.loadmat(path_data)
        test_dataset = test_dataset_mat['data_array']  # (605, 474)
        test_data = test_dataset[:, :-1]
        test_labels = [int(i) - 1 for i in test_dataset[:, -1:].flatten().tolist()]

        a, b = label0.Sort_HeadTail(train_labels, test_labels)
        train_labels = [i - a[np.where(b == i)[0][0]] for i in train_labels]
        test_labels = [i - a[np.where(b == i)[0][0]] for i in test_labels]

    if switch_dataset == 'Protein194':
        path_data = path_dataset + 'Protein194/Protein194Train'
        train_dataset_mat = sciio.loadmat(path_data)
        train_dataset = train_dataset_mat['data_array']
        train_data = train_dataset[:, :-1]
        train_labels = [int(i) - 1 for i in train_dataset[:, -1:].flatten().tolist()]

        path_data = path_dataset + 'Protein194/Protein194Test'
        test_dataset_mat = sciio.loadmat(path_data)
        test_dataset = test_dataset_mat['data_array']
        test_data = test_dataset[:, :-1]
        test_labels = [int(i) - 1 for i in test_dataset[:, -1:].flatten().tolist()]

        a, b = label0.Sort_HeadTail(train_labels, test_labels)
        train_labels = [i - a[np.where(b == i)[0][0]] for i in train_labels]
        test_labels = [i - a[np.where(b == i)[0][0]] for i in test_labels]

    # =====================================================================
    if image_path_test == 'None':  # split root to train and test
        label = [i[1] for i in image_path]
        _, c = utility0.Unique1(label, False)
        kind = 'balanced'  # 'balanced', 'longtailed.  perclass_test
        ratio_train_test = 6  # test = 1 / 6
        if kind == 'balanced':
            perclass_test = [min(c) // ratio_train_test] * len(c)
        elif kind == 'longtailed':
            perclass_test = [i // ratio_train_test for i in c]

        image_path_test, index_test = [], []
        perclass_test1 = perclass_test.copy()
        for i in range(len(image_path)):
            if perclass_test1[image_path[i][1]] > 0:
                image_path_test.append(image_path[i])
                index_test.append(i)
                perclass_test1[image_path[i][1]] -= 1
        index_test.reverse()
        for i in index_test:
            image_path.pop(i)

    if 'IR' in switch_imbalance:
        label = [i[1] for i in image_path]
        _, perclass = utility0.Unique1(label, False)
        perclass = [min(perclass)] * len(perclass)
        perclass = LongTailDistribution(perclass, switch_imbalance)
        selected, perclass1 = [], perclass.copy()
        for i in range(len(label)):
            if perclass1[label[i]] > 0:
                selected.append(i)
                perclass1[label[i]] -= 1
        image_path = [image_path[i] for i in selected]

    # relation_semantic ===================================================
    flag_relation_semantic = False
    if flag_relation_semantic:
        list1 = []  # fine class
        f_label_list = []
        for i in range(len(image_path)):
            a = image_path[i]
            name = a[0]
            f_label = a[1]
            c_label = a[2]

            if len(list1) < 1:
                list1.append([name, f_label, c_label])
                f_label_list.append(f_label)
            else:
                if f_label not in f_label_list:
                    list1.append([name, f_label, c_label])
                    f_label_list.append(f_label)
        # print(list1)

        l1_c = []
        for i in range(len(list1)):
            c = list1[i][2]

            if len(l1_c) < 1:
                l1_c.append(c)
            else:
                if c not in l1_c:
                    l1_c.append(c)
        # print(max(l1_c), min(l1_c))

        for j in range(len(l1_c)):
            print('c:', l1_c[j])
            for i in range(len(list1)):
                name = list1[i][0].split('/height2_')[1].rsplit('/', 1)[0]
                name_f = name.split('/')[1]
                name_c = name.split('/')[0]
                f = list1[i][1]
                c = list1[i][2]
                # print(list1[i])
                # print(f, c, name_f, name_c)

                if c == l1_c[j]:
                    print(f, name_f, c, name_c)
        e

    # create dataset ===================================================
    path_create_dataset = f'{path_all_dataset}ZhaoWei/{switch_dataset}/'
    path_create_dataset = False
    if path_create_dataset:
        def f1_create_dataset(path, image_path):
            shutil.rmtree(path) if os.path.exists(path) else ''
            for i in trange(len(image_path)):
                fine = image_path[i][0].rsplit('/', 2)[1]
                path_fine = path + f'{fine}/'
                os.makedirs(path_fine) if not os.path.exists(path_fine) else ''
                image = image_path[i][0].rsplit('/', 1)[1]
                path_image = path_fine + image

                shutil.copyfile(image_path[i][0], path_image)

        f1_create_dataset(path_create_dataset + 'train/', image_path)
        f1_create_dataset(path_create_dataset + 'test/', image_path_test)
        e

    return image_path, image_path_test, transform, transform_test, num_cls, num_cls_c, relation, eval_tree


def IndexFromDataset_traintest(dataset, num_perclass, rate2=5, rate3=0, shuffle=False):
    # dataset: datasets.CIFAR100
    # num_perclass: [600] * 100.  It must contain entire dataset, because 'if minor else major'
    # rate2=5: train / test = 5 / 1
    # rate3=9: test / val = 9 / 1

    length = dataset.__len__()
    num_perclass = num_perclass.copy()
    # Suppose that all classes have the same number of test samples
    num_test = num_perclass[0]

    if rate3 == 0:  # no val, split to 2 parts
        major = []
        minor = []
        indices = list(range(length))
        if shuffle:
            np.random.shuffle(indices)
        for i in range(length):
            index = indices[i]
            _, label = dataset.__getitem__(index)
            if num_perclass[label] > (rate2 / (rate2 + 1) * num_test):
                minor.append(index)
                num_perclass[label] -= 1
            else:
                major.append(index)
                num_perclass[label] -= 1

        return major, minor, 'no val'
    else:  # split to 3 parts, i.e., val, test and train
           # split dataset to train an test, then split test to test(true) and val
        major = []
        minor_minor = []
        minor_major = []
        indices = list(range(length))
        if shuffle:
            np.random.shuffle(indices)
        for i in range(length):
            index = indices[i]
            _, label = dataset.__getitem__(index)
            if num_perclass[label] > ((rate2/(rate2+1) + 1/(rate2+1) * rate3/(rate3+1)) * num_test):
                minor_minor.append(index)
                num_perclass[label] -= 1
            elif num_perclass[label] > (rate2 / (rate2 + 1) * num_test):
                minor_major.append(index)
                num_perclass[label] -= 1
            else:
                major.append(index)
                num_perclass[label] -= 1

        return major, minor_major, minor_minor


def IndexFromDataset_imbalanced(dataset, num_perclass):
    length = dataset.__len__()
    num_perclass = num_perclass.copy()
    selected_list = []
    indices = list(range(length))

    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_perclass[label] > 0:
            selected_list.append(index)
            num_perclass[label] -= 1

    return selected_list


def IndexFromDataset_imbalanced1(dataset, num_perclass):
    targets = np.array(dataset.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    class_indices = [np.where(targets == i)[0] for i in range(len(classes))]
    indices = [class_index[:num_class] for class_index, num_class in zip(class_indices, num_perclass)]
    indices = np.hstack(indices)

    return indices


def DatasetFromIndex(dataset, indices):
    dataset.data = dataset.data[indices]
    dataset.targets = np.array(dataset.targets)[indices]

    return dataset


def LongTailDistribution(num_perclass_train, switch_imbalance='IR100'):
    ratio = int(switch_imbalance.rsplit('IR', 1)[1])

    max_num = max(num_perclass_train)
    class_num = len(num_perclass_train)
    mu = np.power(1 / ratio, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        # print('%.2f' % np.power(mu, i), end=' ')
        # 1.00 0.60 0.36 0.22 0.13 0.08 0.05 0.03 0.02 0.01
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def NumPerclass(train_labels, test_labels, switch_imbalance):
    if test_labels == 'traintest':
        a, b = np.unique(train_labels, return_counts=True)
        # print(a)
        # print(b)
        num_cls = len(a)
        percls_dataset = b.tolist()
        # print(num_perclass_dataset, num_classes)
        # for i in range(len(num_perclass_dataset) - 1):
        #     if num_perclass_dataset[i] < num_perclass_dataset[i + 1]:
        #         print('no sort')
        num_headclass = percls_dataset[0]
        num_tailclass = percls_dataset[num_cls - 1]
        # if num_headclass > 2 * num_tailclass:
        #     print('the original dataset is imbalanced')

        ratio_test_train = 5  # train / test = 5 / 1
        percls = [int(i * ratio_test_train / (1 + ratio_test_train)) for i in percls_dataset]
        percls_test = [i - j for i, j in zip(percls_dataset, percls)]
    else:
        percls_dataset, ratio_test_train = '', ''

        a, b = np.unique(train_labels, return_counts=True)
        # print(a)
        # print(b)
        num_cls = len(a)
        percls = b.tolist()
        # print(num_perclass_dataset, num_classes)
        # for i in range(len(num_perclass_dataset) - 1):
        #     if num_perclass_dataset[i] < num_perclass_dataset[i + 1]:
        #         print('no sort')
        num_headclass = percls[0]
        num_tailclass = percls[num_cls - 1]
        # if num_headclass > 2 * num_tailclass:
        #     print('the original dataset is imbalanced')

        a, b = np.unique(test_labels, return_counts=True)
        # print(a)
        # print(b)
        num_cls = len(a)
        percls_test = b.tolist()
        # print(num_perclass_dataset, num_classes)
        # for i in range(len(num_perclass_dataset) - 1):
        #     if num_perclass_dataset[i] < num_perclass_dataset[i + 1]:
        #         print('no sort')

    # make balanced train to imbalanced train, but keep test balanced
    if 'IR' in switch_imbalance:
        percls = LongTailDistribution(percls, switch_imbalance)

    return percls_dataset, percls, percls_test, ratio_test_train


def NumPerclass_Coarse(num_cls, num_cls_c, relation, percls):
    percls_c = [0] * num_cls_c
    for i in range(num_cls):
        c = relation[i]
        n = percls[i]
        percls_c[c] += n

    # print(sum(num_perclass_train_c), num_perclass_train_c)
    return percls_c


def DatasetIndex(dataset, num_perclass):
    # Input: a dataset (e.g., cifar100), num_perclass: a list of numbers per class
    # Output: a list of imbalanced indices from a dataset.
    length = dataset.__len__()  # 50000
    num_perclass = list(num_perclass)
    selected_list = []
    indices = list(range(0, length))  # [0, 1, 2, ..., 49999]

    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_perclass[label] > 0:
            selected_list.append(index)
            num_perclass[label] -= 1

    return selected_list


def DataloaderFromDataset(train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resample,
                          defer_epoch, batch_size, epoch):
    num_workers, pin_memory, batch_size_test, shuffle_test = 4, True, 100, False

    if switch_resample == 'RS' and epoch == defer_epoch:
        train_idx = comparison0.IndexFromDataset_resample(train_dataset, num_perclass_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=pin_memory, sampler=WeightedRandomSampler(train_idx, len(train_idx)))
    elif switch_resample == 'SMOTE' and epoch == defer_epoch:
        aug_data, aug_label = comparison0.SMOTE(train_dataset.image, train_dataset.label,
                                    len(num_perclass_train), num_perclass_train[0])
        train_dataset.image = np.concatenate((train_dataset.image, aug_data), axis=0)
        train_dataset.label = np.concatenate((train_dataset.label, aug_label), axis=0)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)

    # label_custom.LoaderlMethod1(train_loader)

    ratio_val_test = 0
    if ratio_val_test != 0:
        test_idx, val_idx, _ = IndexFromDataset_traintest(test_dataset, num_perclass_test, ratio_val_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, num_workers=num_workers,
                                 pin_memory=pin_memory, sampler=SubsetRandomSampler(test_idx))
        val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, num_workers=num_workers,
                                pin_memory=pin_memory, sampler=SubsetRandomSampler(val_idx))
    else:
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=shuffle_test,
                                 num_workers=num_workers, pin_memory=pin_memory)
        val_loader = 'no val'

    # label_custom.LoaderlMethod1(test_loader)
    # label_custom.LoaderlMethod1(val_loader)

    # label_custom.Repetition(_, _, train_loader, test_loader)

    return train_loader, test_loader, val_loader


def Tree(path_data):
    dataset_mat = sciio.loadmat(path_data)  # <class 'dict'>
    dataset = dataset_mat['data_array']
    # print(type(dataset), dataset.shape)  # <class 'numpy.ndarray'> (60000, 4097)
    tree = dataset_mat['tree']
    # print(tree.shape)  # (121, 2)
    # [[105   2]
    #  [102   2]
    #  [115   2]
    # ...
    #  [121   1]
    #  [121   1]
    #  [  0   0]]

    # data = dataset[:, :-1]
    # label = dataset[:, -1:]
    # print(data.shape, data.max(), data.min())  # (60000, 4096) 31.5712 0.0
    # print(label.shape, label.max(), label.min())  # (60000, 1) 100.0 1.0

    return dataset, tree


