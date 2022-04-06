import sys, shutil, math, csv, os, numpy as np, random, torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('TkAgg')  # linux
import matplotlib.pyplot as plt

import label0


# ==========================================================================================
def Unique1(labels, isPrint=True, isDraw=False):
    a, b = np.unique(np.array(labels), return_counts=True)

    if isPrint:
        print(a, 'len:', len(a), 'sum:',sum(a))
        print(b, 'len:', len(b), 'sum:',sum(b))

    if isDraw:
        import matplotlib
        matplotlib.use('TkAgg')  # linux
        import matplotlib.pyplot as plt
        x = np.array([int(i) for i in a])
        y = np.array([int(i) for i in b])
        plt.plot(x, y)
        plt.show()
        plt.close()

    return a, b


# ==========================================================================================
def RandomName(length=4):
    name = ''
    for i in range(length):
        # ran1 = str(random.randint(0,9))  # 0~9之间的数字
        # ran2 = chr(random.randint(65,90))  # A~Z的字母
        ran3 = chr(random.randint(97, 122))  # a~z的字母
        # r = random.choice([ran1, ran2, ran3])  # 从ran1, ran2, ran3中选出一个字符

        name += ran3
    return name


def FixSeed(SEED):
    torch.manual_seed(SEED)  # cpu
    torch.cuda.manual_seed(SEED)  # gpu
    torch.cuda.manual_seed_all(SEED)  # multi-gpu
    np.random.seed(SEED)  # numpy
    random.seed(SEED)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # True: the speed of network is quicker than False, but failing to reproduce
    # def worker_init_fn(worker_id):  # num_workers
    #     np.random.seed(SEED + worker_id)


# ==========================================================================================
def CreateLog(random_name, directory_log, isTest):
    shutil.rmtree(directory_log) if os.path.exists(directory_log) else ''  # os.remove()  删除非空文件夹会拒绝访问
    os.makedirs(directory_log) if not os.path.exists(directory_log) else ''
    txt_name = f'log_{random_name}.txt'
    directory_txt = directory_log + txt_name
    os.remove(directory_txt) if os.path.exists(directory_txt) else ''
    if not isTest:
        csv_name = f'log_{random_name}.csv'
        directory_csv = directory_log + csv_name
        os.remove(directory_csv) if os.path.exists(directory_csv) else ''

        csv_header = ['l', 'l_c', 'l_f', 'acc_c', 'acc_f', 'acc5_f', 'fh_f', 'tie_f',
                      'learn_rate', 'transfer_epoch', 'cluster_epoch',
                      'l_train', 'l_c_train', 'l_f_train', 'acc_c_train', 'acc_f_train']
        with open(directory_csv, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(csv_header)

        csv_name_cf_matrix = 'cf_matrix.csv'
        directory_csv_cf_matrix = directory_log + csv_name_cf_matrix
        os.remove(directory_csv_cf_matrix) if os.path.exists(directory_csv_cf_matrix) else ''

    return directory_txt, directory_csv, csv_header, directory_csv_cf_matrix


def SaveLog(str, file=False, isPrint=False, isLineWrap=True):
    if file != False:
        txt_writer = open(file, 'a')
        txt_writer.write('%s' % str + '\n') if isLineWrap  else txt_writer.write('%s' % str)
        txt_writer.flush()

    if isPrint:
        print('%s' % str) if isLineWrap else print('%s' % str, end='')
    sys.stdout.flush()


def SaveCsv(reserve, file_csv, csv_header=False):
    if csv_header:
        content = []
        for i in csv_header:
            for r in list(reserve.items()):
                key, value = r[0], r[1]
                if i == key:
                    if i != 'learn_rate':
                        value = round(value, 4)
                    content.append(value)
                    break
    else:
        content = reserve

    with open(file_csv, 'a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(content)


def SaveModel(model, epoch, name, model_epoch, switch, directory_log):
    model_name = f'model_{name}_ep{epoch}_sw{switch}.pkl'

    if model_epoch == -1:  # only save one model
        for i in os.listdir(directory_log):
            if f'model_{name}' in i:
                os.remove(directory_log + i)
    else:  # save multiple models
        for i in os.listdir(directory_log):
            if 'model' in i:
                isRemove_i = True
                for j in model_epoch:
                    if f'ep{j}' in i:
                        isRemove_i = False

                if isRemove_i:
                    os.remove(directory_log + i)

    torch.save(model.state_dict(), directory_log + model_name)


def SaveBestResult(reserve, cf_matrix, model, epoch, best_result, switch, directory_log):
    if reserve['acc_c'] > best_result['best_acc_c'][0]:
        best_result['best_acc_c'][0] = reserve['acc_c']
        best_result['best_acc_c'][1] = epoch

    if reserve['acc_f'] > best_result['best_acc_f'][0]:
        best_result['best_acc_f'][0] = reserve['acc_f']
        best_result['best_acc_f'][1] = epoch
        SaveModel(model, best_result['best_acc_f'][1], 'acc', -1, switch, directory_log)
        best_result['best_cf_matrix'] = cf_matrix
    if reserve['acc5_f'] > best_result['best_acc5_f'][0]:
        best_result['best_acc5_f'][0] = reserve['acc5_f']
        best_result['best_acc5_f'][1] = epoch
    if reserve['fh_f'] > best_result['best_fh_f'][0]:
        best_result['best_fh_f'][0] = reserve['fh_f']
        best_result['best_fh_f'][1] = epoch
        SaveModel(model, best_result['best_fh_f'][1], 'fh', -1, switch, directory_log)
    if reserve['tie_f'] < best_result['best_tie_f'][0]:
        best_result['best_tie_f'][0] = reserve['tie_f']
        best_result['best_tie_f'][1] = epoch
        SaveModel(model, best_result['best_tie_f'][1], 'tie', -1, switch, directory_log)

    return best_result


def LoadModel_test(model, directory_log):
    a = directory_log.rsplit('sw')[0]
    b = directory_log.rsplit('sw')[1].split('/')[0]
    for i in os.listdir(a):
        if b in i:
            c = i

    d = a + c + '/'
    for i in os.listdir(d):
        if 'log' in i and 'txt' in i:
            name = i.split('log_')[1].split('.txt')[0]

    for i in os.listdir(d):
        if 'model_acc' in i:
            file_model = torch.load(d + i)

    model.load_state_dict(file_model)  # 加载保存的模型

    return d, name


def LoadModel_pre_training(model, file, dir1, switch, str_ep, int_ep, directory_log0):
    record_model = []
    for i in os.listdir(directory_log0):
        if dir1 == i:
            dir = os.path.join(directory_log0, i)
            for j in os.listdir(dir):
                if switch in j:
                    dir = os.path.join(dir, j)
                    for k in os.listdir(dir):
                        if 'model' in k:
                            record_model.append(k)

    if int_ep == -1:
        if len(record_model) == 1:
            for i in record_model:
                file_model = torch.load(os.path.join(dir, i))
        elif len(record_model) == 2:
            flag_max = 0
            for i in record_model:
                i_epoch = int(i.split('ep')[1].split('_sw')[0])
                if flag_max == 0:
                    max = i_epoch
                    flag_max = 1
                else:
                    if max < i_epoch:
                        min = max
                        max = i_epoch
                    else:
                        min = i_epoch

            if str_ep == 'final':
                t = str(max)
            elif str_ep == 'best':
                t = str(min)

            for i in record_model:
                if 'ep' + t in i:
                    file_model = torch.load(os.path.join(dir, i))
                    break
    else:
        for i in record_model:
            if f'ep{int_ep}' in i:
                file_model = torch.load(os.path.join(dir, i))
                break

    SaveLog('LoadModel: {}'.format(os.path.join(dir, i)), file)
    model.load_state_dict(file_model)  # 加载保存的模型


# ==========================================================================================
def VisualList(a, file, column_number=15, word_max_width=0, title=''):
    if type(a) is torch.Tensor:
        a = a.numpy()
    if type(a) is np.ndarray:
        a = a.flatten()
        a = a.tolist()
    # transform input into string
    a = [str(i) for i in a]
    len_a = len(a)
    len_last_rows = 0

    flag_over = 0
    OmitGrid = True

    margin_left = 0
    margin_right = 0
    if word_max_width == 0:  # automatically adjust width according to input
        for i in range(len(a)):
            if word_max_width < len(a[i]):
                word_max_width = len(a[i])
    word_width = word_max_width + margin_left
    width_a = '{' + f':>{word_width}' + '}'
    width_a1 = '{' + f':<{word_width}' + '}'

    row_number = 999  # 999 is a relatively big number
    word_width_row = len(str(row_number)) + margin_left  # leftmost column
    width_row = '{' + f':<{word_width_row}' + '}'

    style_grid = 0
    if style_grid == 0:
        # │ - ┼ ┌ ┐ └ ┘ ├ ┤ ┬ ┴
        grid_corner_lt = '┌'
        grid_corner_lb = '└'
        grid_corner_rt = '┐'
        grid_corner_rb = '┘'

        grid_edge_l = '├'
        grid_edge_r = '┤'
        grid_edge_t = '┬'
        grid_edge_b = '┴'

        grid_row = '─'
        grid_column = '│'
        grid_cross = '┼'
    if style_grid == 1:
        # ┃ ━ ╋ ┏ ┓ ┗ ┛ ┣ ┫ ┳ ┻
        grid_corner_lt = '┏'
        grid_corner_lb = '┗'
        grid_corner_rt = '┓'
        grid_corner_rb = '┛'

        grid_edge_l = '┣'
        grid_edge_r = '┫'
        grid_edge_t = '┳'
        grid_edge_b = '┻'

        grid_row = '━'
        grid_column = '┃'
        grid_cross = '╋'
    if style_grid == 2:
        grid_corner_lt = '┌'
        grid_corner_lb = '└'
        grid_corner_rt = '┐'
        grid_corner_rb = '┘'

        grid_edge_l = '├'
        grid_edge_r = '┤'
        grid_edge_t = '┬'
        grid_edge_b = '┴'

        grid_row = '┄'
        grid_column = '┊'
        grid_cross = '┼'

    # ==========================================================================
    # grid_top
    SaveLog(grid_corner_lt, file, False, False)
    SaveLog(word_width_row * grid_row, file, False, False)
    SaveLog(margin_right * grid_row, file, False, False)
    SaveLog(grid_edge_t, file, False, False)
    for column_i in range(column_number):
        SaveLog(word_width * grid_row, file, False, False)
        SaveLog(margin_right * grid_row, file, False, False)
        if column_i < column_number - 1:
            SaveLog(grid_edge_t, file, False, False)
        else:
            SaveLog(grid_corner_rt, file, False, True)

    # number_top
    SaveLog(grid_column, file, False, False)
    SaveLog(width_row.format(title), file, False, False)
    SaveLog(margin_right * ' ', file, False, False)
    SaveLog(grid_column, file, False, False)
    for column_i in range(column_number):
        SaveLog(width_a1.format(column_i), file, False, False)
        SaveLog(margin_right * ' ', file, False, False)
        if column_i < column_number - 1:
            SaveLog(grid_column, file, False, False)
        else:
            SaveLog(grid_column, file, False, True)
    # ==========================================================================
    for row_i in range(row_number):
        # grid_centre
        if OmitGrid == True:
            if row_i == 0:
                SaveLog(grid_edge_l, file, False, False)
                SaveLog(word_width_row * grid_row, file, False, False)
                SaveLog(margin_right * grid_row, file, False, False)
                SaveLog(grid_cross, file, False, False)
                for column_i in range(column_number):
                    SaveLog(word_width * grid_row, file, False, False)
                    SaveLog(margin_right * grid_row, file, False, False)
                    if column_i < column_number - 1:
                        SaveLog(grid_cross, file, False, False)
                    else:
                        SaveLog(grid_edge_r, file, False, True)
        else:
            SaveLog(grid_edge_l, file, False, False)
            SaveLog(word_width_row * grid_row, file, False, False)
            SaveLog(margin_right * grid_row, file, False, False)
            SaveLog(grid_cross, file, False, False)
            for column_i in range(column_number):
                SaveLog(word_width * grid_row, file, False, False)
                SaveLog(margin_right * grid_row, file, False, False)
                if column_i < column_number - 1:
                    SaveLog(grid_cross, file, False, False)
                else:
                    SaveLog(grid_edge_r, file, False, True)

        # number_centre
        SaveLog(grid_column, file, False, False)
        SaveLog(width_row.format(len_last_rows), file, False, False)  # len_last_rows -- row_i
        SaveLog(margin_right * ' ', file, False, False)
        SaveLog(grid_column, file, False, False)
        for column_i in range(column_number):
            if column_i + len_last_rows < len_a:
                SaveLog(width_a.format(a[column_i + len_last_rows]), file, False, False)
                SaveLog(margin_right * ' ', file, False, False)
                SaveLog(grid_column, file, False, False)
                if column_i == column_number - 1:
                    len_last_rows += column_i + 1
                    SaveLog('', file, False, True)
            else:
                flag_over = 1

        if len_last_rows == len(a):
            break
        if flag_over == 1:
            SaveLog('', file, False, True)
            break
    # ==========================================================================
    # grid_bottom
    SaveLog(grid_corner_lb, file, False, False)
    SaveLog(word_width_row * grid_row, file, False, False)
    SaveLog(margin_right * grid_row, file, False, False)
    SaveLog(grid_edge_b, file, False, False)
    for column_i in range(column_number):
        SaveLog(word_width * grid_row, file, False, False)
        SaveLog(margin_right * grid_row, file, False, False)
        if column_i < column_number - 1:
            SaveLog(grid_edge_b, file, False, False)
        else:
            SaveLog(grid_corner_rb, file, False, True)


def ListCombine(a, b, c='/'):  # aim to VisualList()
    a = [str(i) for i in a]
    b = [str(i) for i in b]
    a_b = []
    record = []
    for i, j in zip(range(len(a)), range(len(b))):
        if a[i] == b[j]:
            t = a[i]
            record.append(i)
        else:
            t = a[i] + c + b[j]
        a_b.append(t)

    return a_b, record


# ==========================================================================================
def PrintOnRowWidth(str1, row_width, file):
    seg = math.ceil(len(str1) / row_width)  # 每行输出row_length 个向上取整有seg行，
    for i in range(seg):
        startpoint = row_width * i  # 每行的索引点
        SaveLog(str1[startpoint: startpoint + row_width], file, False)  # 索引字符串


def VisualRelation_fine_to_coarse(name_c_classes, num_perclass_train_c, name_f_classes, num_perclass_train,
                                  relation_f, file, isChinese=False):
    relation_c_array = label0.RelationCoarseArray(len(name_c_classes), relation_f)
    for i in range(len(name_c_classes)):
        relation_c_single = label0.RelationCoarse_Single(i, relation_c_array)
        d = []
        for j in relation_c_single:
            d.append(num_perclass_train[j])

        if isChinese:
            # {:\u3000<12}  纯中文输出对齐，不能中文混合英文（数字）
            SaveLog('{:\u3000<12}({:<5})  {}[{}]'.format(
                name_c_classes[i], num_perclass_train_c[i], [name_f_classes[i] for i in relation_c_single], d),
                file, False)
        else:
            SaveLog('{:<3}({:<5})  {}[{}]'.format(
                name_c_classes[i], num_perclass_train_c[i], [name_f_classes[i] for i in relation_c_single], d),
                file, False)


def AccToString(a, decimals=4):
    a = round(a, decimals)
    a = str(a)
    b = ''
    for i in range(decimals + 2):
        if i > 1:
            try:
                b += a[i]
            except:
                b += '0'

    return b


# ==========================================================================================
def LearnRate(epoch, stage_length, switch_lr):
    lr_init = 0.1

    if switch_lr == 0:  # LDAM: stage_length is 200
        if epoch < 5:
            lr = lr_init * (epoch + 1) / 5
        elif stage_length - 40 <= epoch < stage_length - 20:
            lr = lr_init * 0.01
        elif stage_length - 20 <= epoch:
            lr = lr_init * 0.0001
        else:
            lr = lr_init
    elif switch_lr == 1:
        if epoch < 5:
            lr = lr_init * (epoch + 1) / 5
        elif 0.8 * stage_length <= epoch < 0.9 * stage_length:
            lr = lr_init * 0.01
        elif 0.9 * stage_length <= epoch:
            lr = lr_init * 0.0001
        else:
            lr = lr_init
    elif switch_lr == 2:
        if stage_length - 40 <= epoch < stage_length - 20:
            lr = lr_init * 0.01
        elif stage_length - 20 <= epoch:
            lr = lr_init * 0.0001
        else:
            lr = lr_init
    elif switch_lr == 3:
        if 0.8 * stage_length <= epoch < 0.9 * stage_length:
            lr = lr_init * 0.01
        elif 0.9 * stage_length <= epoch:
            lr = lr_init * 0.0001
        else:
            lr = lr_init
    elif switch_lr == 4:
        lr = lr_init * 0.0001
    elif switch_lr == 5:
        if epoch < 5:
            lr = lr_init * (epoch + 1) / 5
        else:
            lr = lr_init
    elif switch_lr == 6:
        print()
    elif switch_lr == 7:
        print()
    elif switch_lr == 8:
        if epoch < 60:
            lr = 0.1
        elif 60 <= epoch < 120:
            lr = 0.02
        elif 120 <= epoch < 160:
            lr = 0.004
        else:
            lr = 0.008
    elif switch_lr == 9:
        lr = lr_init * 0.01
    else:
        raise ValueError(switch_lr)

    return lr


def AdjustLearnRate(epoch, optimizer, switch_lr, coarse_epoch, hier_epoch, fine_epoch):
    if epoch < coarse_epoch:
        stage_length = coarse_epoch
        param_lr = LearnRate(epoch, stage_length, switch_lr)
    elif coarse_epoch <= epoch < coarse_epoch + hier_epoch:
        epoch -= coarse_epoch
        stage_length = hier_epoch
        param_lr = LearnRate(epoch, stage_length, switch_lr)
    else:
        epoch -= coarse_epoch + hier_epoch
        stage_length = fine_epoch
        param_lr = LearnRate(epoch, stage_length, switch_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_lr

    return param_lr


def VisualLearningRate(file_txt, file_csv, isPrint=False):  # list
    record_lr = ColumnFromCsv('learn_rate', file_csv)
    record_lr = np.array([float(i) for i in record_lr])

    visual_lr = np.array([[0.0, 0.0]])  # [0, 0]: [epoch, lr]
    i_visual = 0
    for epoch in range(len(record_lr)):
        if epoch == 0:
            visual_lr[i_visual, 1] = record_lr[epoch]
        else:
            if record_lr[epoch] == record_lr[epoch - 1]:
                visual_lr[i_visual, 0] += 1
            else:
                row_visual = np.array([epoch, record_lr[epoch]])
                visual_lr = np.row_stack((visual_lr, row_visual))
                i_visual += 1

    width = 8
    SaveLog('Epoch{}LR'.format(' ' * width), file_txt, isPrint)

    visual_lr_str = visual_lr.astype(np.str_)
    for i_visual in range(len(visual_lr)):
        if i_visual == 0:
            visual_lr_str[i_visual, 0] = f'0 - {int(visual_lr[i_visual, 0])}:'
        else:
            visual_lr_str[i_visual, 0] = f'{int(visual_lr[i_visual - 1, 0] + 1)} - {int(visual_lr[i_visual, 0])}:'

        str_epoch = visual_lr_str[i_visual, 0]
        str_lr = visual_lr_str[i_visual, 1]
        b = len('Epoch') + width - len(str_epoch)
        SaveLog('{}{}{}'.format(str_epoch, ' ' * b, str_lr), file_txt, isPrint)


# ==========================================================================================
def ColumnFromCsv(header, file_csv):
    with open(file_csv, 'r') as f:
        reader = csv.DictReader(f)
        column = [row[header] for row in reader]

    return column


def DrawOnCsv_loss(flag_loss , file_csv, directory_log, id_name=False):
    L_ = ColumnFromCsv('L_', file_csv)  # str
    L = ColumnFromCsv('L', file_csv)
    L_c_ = ColumnFromCsv('L_c_', file_csv)
    L_c = ColumnFromCsv('L_c', file_csv)
    L_f_ = ColumnFromCsv('L_f_', file_csv)
    L_f = ColumnFromCsv('L_f', file_csv)

    L = np.array([float(i) for i in L])
    L_ = np.array([float(i) for i in L_])
    L_c = np.array([float(i) for i in L_c])
    L_c_ = np.array([float(i) for i in L_c_])
    L_f = np.array([float(i) for i in L_f])
    L_f_ = np.array([float(i) for i in L_f_])

    lr = ColumnFromCsv('lr', file_csv)
    lr = np.array([float(i) for i in lr])

    x = np.linspace(0, len(L_) - 1, num=len(L_), dtype=int)

    # create a new canvas, otherwise you will put more than one painting on the same canvas
    plt.figure(figsize=(6, 6), dpi=300)

    if flag_loss == 'l':
        plt.plot(x, L_, color='black', linestyle='--')  # '-', '--', '-.', ':'
        plt.plot(x, L, label='L', color='black', linestyle='-')
        plt.plot(x, L_c_, color='darkgray', linestyle='--')
        plt.plot(x, L_c, label='L_c', color='darkgray', linestyle='-')
        plt.plot(x, L_f_, color='orange', linestyle='--')
        plt.plot(x, L_f, label='L_f', color='orange', linestyle='-')
    if flag_loss == 'l_c':
        plt.plot(x, L_c_, color='darkgray', linestyle='--')
        plt.plot(x, L_c, label='L_c', color='darkgray', linestyle='-')
    if flag_loss == 'l_f':
        plt.plot(x, L_f_, color='orange', linestyle='--')
        plt.plot(x, L_f, label='L_f', color='orange', linestyle='-')

    plt.plot(x, lr, label='lr', color='red', linestyle=':')

    plt.yticks(np.linspace(0.0, 5.0, num=26))
    plt.grid()
    # plt.ylabel('')
    plt.xlabel('Epoch')
    # plt.title('')
    plt.legend(loc='lower left')  # 注释栏

    plot_name = f'plot_loss_{id_name}' if id_name != False else 'plot_loss'
    plt.savefig(directory_log + plot_name)  # 覆盖同名文件
    # plt.show()
    plt.close()


# ==========================================================================================
def tree_ANcestor(tr, nd):
    #  tree: 二维数组   node：节点  默认根节点为0

    A = [nd]  # 存储
    nd_ = tr[nd - 1]  # python 从 0 开始算，而结点从 1 开始算

    while nd_ > 0:
        A.append(nd_)
        nd_ = tr[nd_ - 1]  # 找父结点

    return A


def EvaHier_TreeInducedError(tr, p_nd, r_nd, switch_dataset):
    # 举例: 真实标签为3，预测标签为 4
    # tr = [0, 1, 1, 3, 2, 3, 6, 6]  # 索引的父结点（从上到下，从左到右）
    # print(EvaHier_TreeInducedError(tr, [4], [6]))
    # print(EvaHier_TreeInducedError(tr, [4, 1, 2, 3, 2, 5, 5], [6, 2, 3, 4, 2, 4, 6]))

    if 'VOC' in switch_dataset:
        max1 = 7
    elif 'SUN' in switch_dataset:
        max1 = 6
    else:
        max1 = 4

    tr = [i+1 for i in tr]

    # TIE = 0
    # for i in range(len(p_nd)):
    #     r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
    #     p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
    #     b = list(set(r_anc).difference(set(p_anc)))  # 取 r_anc 与 p_anc 的差集
    #     c = list(set(p_anc).difference(set(r_anc)))  # 取 p_anc 与 r_anc 的差集
    #     TIE += len(b + c) / max1
    # TIE = TIE / len(p_nd)

    a, _ = Unique1(r_nd, False)
    cls_hit_TIE = [0] * (max(a) + 1)
    for i in range(len(p_nd)):
        r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
        p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
        b = list(set(r_anc).difference(set(p_anc)))  # 取 r_anc 与 p_anc 的差集
        c = list(set(p_anc).difference(set(r_anc)))  # 取 p_anc 与 r_anc 的差集

        TIE = len(b + c) / max1
        cls_hit_TIE[r_nd[i]] += TIE

    return np.array(cls_hit_TIE)


def EvaHier_HierarchicalPrecisionAndRecall(tr, p_nd, r_nd):
    tr = [i+1 for i in tr]

    # sum_PH, sum_RH, sum_FH = 0, 0, 0
    # length = len(p_nd)
    # for i in range(length):
    #     r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
    #     p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
    #     b = [x for x in r_anc if x in p_anc]  # 取 r_anc 与 p_anc 的交集
    #
    #     PH = len(b) / len(p_anc)
    #     RH = len(b) / len(r_anc)
    #     FH = 2 * PH * RH / (PH + RH)
    #
    #     sum_PH += PH
    #     sum_RH += RH
    #     sum_FH += FH
    #
    # PH = sum_PH / length
    # RH = sum_RH / length
    # FH = sum_FH / length

    a, _ = Unique1(r_nd, False)
    cls_hit_FH = [0] * (max(a) + 1)
    for i in range(len(p_nd)):
        r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
        p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
        b = [x for x in r_anc if x in p_anc]  # 取 r_anc 与 p_anc 的交集

        PH = len(b) / len(p_anc)
        RH = len(b) / len(r_anc)
        FH = 2 * PH * RH / (PH + RH)

        cls_hit_FH[r_nd[i]] += FH

    return np.array(cls_hit_FH)


def ArrayToTree(leaves_array):
    # 对于一个数据集，无论有几层，预先选定某一层作为训练时的粗类。对于一个细类，评价时有可以有多层粗类，但训练时只有一层粗类

    for j in range(1, leaves_array.shape[1]):
        max_j = max(leaves_array[:, j - 1: j].flatten().tolist())
        for i in range(leaves_array.shape[0]):
            if leaves_array[i, j] >= 0:
                leaves_array[i, j] += (max_j + 1)

    eval_tree = [-1] * (np.amax(leaves_array) + 1)  # TIE, FH
    for j in range(1, leaves_array.shape[1]):
        for i in range(leaves_array.shape[0]):
            node = leaves_array[i, j]
            if eval_tree[node] == -1:
                father_node = leaves_array[i, j - 1]
                eval_tree[node] = father_node

    return eval_tree


# ==========================================================================================
def StrToList_relation(str):
    l, t = [], ''
    for i in str:
        if i != '\n' and i != ' ':
            t += i
        else:
            if len(t) > 0:
                l.append(int(t))
                t = ''

    return l


# ==========================================================================================
def FeatureVisual(type, feature, label, device, directory_log, id_name=False):
    feature = torch.tensor(feature)
    if type == 't_SNE':
        x = TSNE(learning_rate=100).fit_transform(feature)
    elif type == 'PCA':
        x = PCA().fit_transform(feature)

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=label)
    plt.xticks([])  # hide tick
    plt.yticks([])

    plot_name = f'visual_{type}_{id_name}' if id_name != False else f'visual_{type}'
    plt.savefig(directory_log + plot_name)  # 覆盖同名文件
    # plt.show()
    plt.close()


# ==========================================================================================
# ==========================================================================================
if __name__ == '__main__':
    print()