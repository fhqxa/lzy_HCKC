import math, numpy as np, torch
from sklearn.cluster import SpectralClustering


def TransferEpoch(epoch, transfer_epoch, coarse_epoch, hier_epoch, fine_epoch, end_epoch,
                  param1_ir):
    if transfer_epoch != 'x':
        flag = 0  # single-task; multi-task
        if flag == 0:
            if epoch < coarse_epoch:  # coarse
                param_transfer_epoch = 1
            elif epoch >= coarse_epoch + hier_epoch:  # fine
                param_transfer_epoch = 0
            else:  # hier
                if transfer_epoch == 5:  # Linear decay
                    param_transfer_epoch = 1 - (epoch - coarse_epoch + 1) / (end_epoch - coarse_epoch - fine_epoch)

    else:
        param_transfer_epoch = 0

    return param_transfer_epoch


def HierLoss(criterion, criterion_c, out, out_c, label, label_c,
             param_transfer_epoch):
    if param_transfer_epoch == 0:  # fine
        l = criterion(out, label)
        l_c = criterion_c(out_c, label_c).detach()
        l0 = l
    elif param_transfer_epoch == 1:  # coarse
        l = criterion(out, label).detach()
        l_c = criterion_c(out_c, label_c)
        l0 = l_c
    else:
        l = criterion(out, label)
        l_c = criterion_c(out_c, label_c)
        l0 = param_transfer_epoch * l_c + (1 - param_transfer_epoch) * ScaleLoss(l_c, l) * l

    return l0, l_c, l


def HierLoss1_epoch(hier_loss1, epoch, coarse_epoch, hier_epoch, fine_epoch, end_epoch):
    if hier_loss1 == 0:  # No transfer
        if epoch < coarse_epoch:
            param_hier_loss1 = 1
        else:  # fine
            param_hier_loss1 = 0
    elif hier_loss1 == 1:
        param_hier_loss1 = 1 - (epoch + 1) / (end_epoch)
    elif hier_loss1 == 2:
        param_hier_loss1 = 1 - ((epoch + 1) / (end_epoch)) ** 2
    elif hier_loss1 == 3:
        param_hier_loss1 = math.cos((epoch + 1) / (end_epoch) * math.pi / 2)
    elif hier_loss1 == 4:
        param_hier_loss1 = 1 / 2 + 1 / 2 * math.cos(math.pi / 2 * (epoch + 1) / (end_epoch))
    elif hier_loss1 == 5:
        param_hier_loss1 = 0.1
    elif hier_loss1 == 6:
        param_hier_loss1 = 0.2
    elif hier_loss1 == 7:
        param_hier_loss1 = 0.3
    elif hier_loss1 == 8:
        param_hier_loss1 = 0.4
    elif hier_loss1 == 9:
        param_hier_loss1 = 0.5

    return param_hier_loss1


def HierLoss1(criterion, criterion_c_semantic, criterion_c_cluster, out, out_c,
              label, label_c_semantic, label_c_cluster,
              param_hier_loss1, param_transfer_epoch):

    if param_transfer_epoch == 0:  # fine
        l = criterion(out, label)

        l_c_semantic = criterion_c_semantic(out_c, label_c_semantic)
        l_c_cluster = criterion_c_cluster(out_c, label_c_cluster)
        l_c = param_hier_loss1 * l_c_semantic + (1 - param_hier_loss1) * l_c_cluster

        l0 = l
    elif param_transfer_epoch == 1:  # coarse
        l = criterion(out, label).detach()

        l_c_semantic = criterion_c_semantic(out_c, label_c_semantic)
        l_c_cluster = criterion_c_cluster(out_c, label_c_cluster)
        l_c = param_hier_loss1 * l_c_semantic + (1 - param_hier_loss1) * l_c_cluster

        l0 = l_c
    else:
        l = criterion(out, label)

        l_c_semantic = criterion_c_semantic(out_c, label_c_semantic)
        l_c_cluster = criterion_c_cluster(out_c, label_c_cluster)
        l_c = param_hier_loss1 * l_c_semantic + (1 - param_hier_loss1) * l_c_cluster

        l0 = param_transfer_epoch * l_c + (1 - param_transfer_epoch) * ScaleLoss(l_c, l) * l

    return l0, l_c, l


def ScaleLoss(num1, num2):
    s1 = int(math.floor(math.log10(num1)))  # num1=100, log10(mum1)=2, s1=2; num1<100, s1=1; num<10, s1=0
    s2 = int(math.floor(math.log10(num2)))
    scale = 10 ** (s1 - s2)
    return scale  # both the value of 'num1' and 'num2' are less than 100, the value of 'scale' always is equal to 1


# ==================================================================
def ClassFeatureVector(data, fine_labels, num_class, device, isTensor=False):
    if isTensor:
        data = torch.from_numpy(data).to(device)
    # print(data.shape)  # torch.Size([10847, 4096])
    class_vec = torch.zeros([num_class, data.shape[1]]).to(device) if isTensor else np.zeros([num_class, data.shape[1]])
    # print(class_vec.shape)  # torch.Size([100, 4096])  100 classes

    for i in range(num_class):
        idx = [j for j, x in enumerate(fine_labels) if x == i]
        # print('i: {}, len(idx): {}'.format(i, len(idx)))
        sigma_cls = torch.zeros(data.shape[1]).to(device) if isTensor else np.zeros(data.shape[1])
        # print(sigma_cls.shape)  # torch.Size([4096])
        for m in range(len(idx)):
            sigma_cls += data[idx[m], :]
            # print(sigma_cls.shape, data[idx[m], :].shape)  # torch.Size([4096]) torch.Size([4096])
        vec = sigma_cls / len(idx)
        class_vec[i] = vec
        # print('vec: {}, {}'.format(vec.shape, vec))  # torch.Size([4096]
    # print('class_vec: {}, {}'.format(class_vec.shape, class_vec))  # torch.Size([100, 4096]),
    return class_vec


def ClusterCoarse(data, fine_labels, num_clusters, device, isTensor=False):
    # fine_labels: [20, 21, 22] → [0, 1, 2]
    a, _ = np.unique(fine_labels, return_counts=True)
    fine_labels = fine_labels.tolist()
    min1 = min(a)
    num_class = len(a)
    if min1 != 0:
        fine_labels = [i - min1 for i in fine_labels]

    class_vec = ClassFeatureVector(data, fine_labels, num_class, device)
    aff_mat = torch.zeros([num_class, num_class]) if isTensor else np.zeros([num_class, num_class])

    for i in range(0, num_class - 1):
        for j in range(i + 1, num_class):
            distance = torch.linalg.norm(class_vec[i] - class_vec[j]) if isTensor \
                else np.linalg.norm(class_vec[i] - class_vec[j])
            aff_mat[i, j] = distance
            aff_mat[j, i] = aff_mat[i, j]
    # aff_mat = normalize_2darray(0, 1, aff_mat)
    beta = 0.1
    aff_mat = torch.exp(-beta * aff_mat / aff_mat.std()) if isTensor else np.exp(-beta * aff_mat / aff_mat.std())
    for i in range(num_class):
        aff_mat[i, i] = 0.0001

    # sc = SpectralClustering(num_clusters, affinity='precomputed', assign_labels='discretize')
    sc = SpectralClustering(num_clusters, n_init=num_clusters, affinity='precomputed', n_neighbors=10,
                            assign_labels='kmeans')

    # print(aff_mat.shape, aff_mat)  # torch.Size([100, 100])
    # groups = sc.fit_predict(aff_mat)
    groups = sc.fit_predict(aff_mat.detach().numpy()) if isTensor else sc.fit_predict(aff_mat)
    if min1 != 0:
        l1 = [i + min1 for i in groups]
        groups = list(range(min1))
        groups.extend(l1)
        groups = np.array(groups)
    # print(groups.shape, groups)  # (100,)
    # [ 6  4  8  8 11  6  9  3  4  5  8  7  3  1  3 18 19  8  1  6  3  1 19  5
    #   7  4  2 15  1  1  1  3  5  8  3 15  8  1  5  4  5  6  9 19  2 15  4  1
    #   9  6  6  6  6 15 17  7  4 15  6  8  3  5  9  7  5  1 11 15 18  2  8  1
    #   4  2  2  3 15  4  4  6  7  5  7 14 19  2  4 11 10  0  0 17  2  7  2 13
    #   0 16 11 12]

    return groups


def FeatureStack(feature_all, label_all, feature, target, tail_f_marks2, cluster_coarse_layer, isTensor=False):
    feature = feature[cluster_coarse_layer]
    if not isTensor:
        feature = feature.cpu().detach().numpy()
    target = target.cpu().numpy()

    idx_tail = np.where(target >= min(tail_f_marks2))[0]
    feature = feature[idx_tail]
    target = target[idx_tail]

    if isinstance(feature_all, str):
        feature_all = feature
        label_all = target
    else:
        feature_all = torch.cat((feature_all, feature)) if isTensor else np.vstack((feature_all, feature))
        label_all = np.hstack((label_all, target))

    return feature_all, label_all


def ClusterEpoch(isStack, epoch, relation_semantic, relation_cluster, feature_cluster, label_cluster,
                 feature, target, tail_f_marks2, model, train_loader, cluster_num, cluster_coarse_layer, device,
                 cluster_epoch, coarse_epoch, hier_epoch):
    if cluster_epoch != 'x':
        # dynamic: dynamic_cluster; fix: fix_cluster; c: coarse_epoch; h: hier_epoch; f: fine_epoch
        # param_cluster_epoch =  0: semantic; 1: dynamic; 2: fix
        flag = 2
        if flag == 2:
            if cluster_epoch == 2:
                if epoch < coarse_epoch:
                    param_cluster_epoch = 0
                else:
                    param_cluster_epoch = 1

        if isStack:  # determine the output of this function
            if param_cluster_epoch == 1:
                feature_cluster, label_cluster = FeatureStack(
                    feature_cluster, label_cluster, feature, target, tail_f_marks2, cluster_coarse_layer)

                return feature_cluster, label_cluster
            else:
                return '', ''
        else:
            if param_cluster_epoch == 1:  # dynamic: update relation_cluster
                if isinstance(relation_cluster, str):
                    model.eval()
                    for data, target in train_loader:
                        data = data.to(device)
                        _, _, feature = model(data)
                        feature_cluster, label_cluster = FeatureStack(
                            feature_cluster, label_cluster, feature, target, tail_f_marks2, cluster_coarse_layer)

                # print(feature_cluster.shape, len(label_feature_cluster))  # (10847, 64, 8, 8) 10847
                feature_cluster = feature_cluster.reshape(feature_cluster.shape[0], -1)
                # print(feature_cluster.shape)  # (10847, 4096)
                relation_cluster = ClusterCoarse(feature_cluster, label_cluster, cluster_num, device)

            return relation_cluster if param_cluster_epoch else relation_semantic, \
                   relation_semantic, relation_cluster, param_cluster_epoch
    else:
        if isStack:
            return '', ''
        else:
            return relation_semantic, relation_semantic, relation_cluster, 0  # 0: semantic


# ============================================================

