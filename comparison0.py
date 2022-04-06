import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def Rebalance(switch_rebalance):
    if switch_rebalance == 'None':
        switch_defer = 0
        switch_resample = 0
        switch_reweight = 0
    elif switch_rebalance == 'RS':
        switch_defer = 0
        switch_resample = 'RS'
        switch_reweight = 0
    elif switch_rebalance == 'SMOTE':
        switch_defer = 0
        switch_resample = 'SMOTE'
        switch_reweight = 0
    elif switch_rebalance == 'RW':
        switch_defer = 0
        switch_resample = 0
        switch_reweight = 'RW'
    elif switch_rebalance == 'CB':
        switch_defer = 0
        switch_resample = 0
        switch_reweight = 'CB'
    elif switch_rebalance == 'DRS':
        switch_defer = 1
        switch_resample = 'RS'
        switch_reweight = 0
    elif switch_rebalance == 'DSMOTE':
        switch_defer = 1
        switch_resample = 'SMOTE'
        switch_reweight = 0
    elif switch_rebalance == 'DRW':
        switch_defer = 1
        switch_resample = 0
        switch_reweight = 'RW'
    elif switch_rebalance == 'DCB':
        switch_defer = 1
        switch_resample = 0
        switch_reweight = 'CB'

    return switch_defer, switch_resample, switch_reweight


def IndexFromDataset_resample(dataset, num_perclass):  # RS
    length = dataset.__len__()
    num_perclass = num_perclass.copy()
    num_perclass1 = num_perclass.copy()

    selected_list = []
    indices = list(range(0, length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_perclass[label] > 0:
            selected_list.append(1 / num_perclass1[label])
            num_perclass[label] -= 1

    # a1 , a2 = np.unique(np.array(selected_list), return_counts=True)
    # print(a1, a2)
    # [0.0002 0.0003 0.0006 0.0009 0.0015 0.0026 0.0043 0.0072 0.012  0.02  ]
    # [5000 2997 1796 1077  645  387  232  139   83   50]
    # print([i * j for i, j in zip(a1, a2)])
    # [1.0, 0.899, 1.0776, 0.969, 0.9675, 1.0062, 0.9976, 1.0008, 0.996, 1.0]
    return selected_list


def SMOTE(data, targets, n_class, n_max):
    aug_data = []
    aug_label = []

    for k in range(1, n_class):
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)
        class_dist = np.zeros((class_len, class_len))

        for i in range(class_len):
            for j in range(class_len):
                class_dist[i, j] = np.linalg.norm(class_data[i] - class_data[j])
        sorted_idx = np.argsort(class_dist)

        for i in range(n_max - class_len):
            lam = np.random.uniform(0, 1)
            row_idx = i % class_len
            col_idx = int((i - row_idx) / class_len) % (class_len - 1)
            new_data = np.round(
                lam * class_data[row_idx] + (1 - lam) * class_data[sorted_idx[row_idx, 1 + col_idx]])

            aug_data.append(new_data.astype('uint8'))
            aug_label.append(k)

    return np.array(aug_data), np.array(aug_label)


def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0., reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction), self.gamma)


class LDAMLoss(nn.Module):
    def __init__(self, device, cls_num_list, weight=None, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.device = device

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor).to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


# ============================================================


