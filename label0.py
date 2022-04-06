import numpy as np

import dataset0


def RelationFine(tree):
    num_f_classes = 0
    num_c_classes = 0
    for node in range(tree.shape[0]):
        layer = tree[node, 1]
        if layer == 2:  # 2: 2th layer, i.e., fine layer
            num_f_classes += 1
        if layer == 1:  # 1: 1th layer, i.e., coarse layer
            num_c_classes += 1

    # index: f; value: c
    # default: 1st fine label is 0, 1st coarse label is 0
    relation_f = [(tree[i, 0] - num_f_classes) for i in range(num_f_classes)]
    relation_f = [i - 1 for i in relation_f]  # 标签转化为从0开始

    return num_f_classes, num_c_classes, relation_f


def RelationFine1(f_labels, c_labels):
    num_f_classes = max(f_labels) - min(f_labels) + 1  # 99 0
    num_c_classes = max(c_labels) - min(c_labels) + 1  # 19 0
    relation_f = [-1] * num_f_classes

    for f, c in zip(f_labels, c_labels):  # <class 'list'>
        for i in range(len(relation_f)):
            if relation_f[f] == -1:
                relation_f[f] = c

    return num_f_classes, num_c_classes, relation_f


def RelationCoarseArray(num_c_classes, relation_f):
    num_f_classes = len(relation_f)
    # 2D: 0index, c; 1index, f
    relation_c_array = np.full((num_c_classes, num_f_classes), -1)

    for c in range(num_c_classes):
        num_perc = 0
        for f in range(num_f_classes):
            if c == relation_f[f]:
                relation_c_array[c, num_perc] = f
                num_perc += 1

    return relation_c_array


def RelationFine_follow(tree):
    num_f_classes = 0
    num_c_classes = 0
    for node in range(tree.shape[0]):
        layer = tree[node, 1]
        if layer == 2:  # 2: 2th layer, i.e., fine layer
            num_f_classes += 1
        if layer == 1:  # 1: 1th layer, i.e., coarse layer
            num_c_classes += 1

    # index: f; value: c
    # default: 1st fine label is 0, 1st coarse label follows the last fine label
    relation_f_follow = [tree[i, 0] for i in range(num_f_classes)]
    relation_f_follow = [i - 1 for i in relation_f_follow]

    return num_f_classes, num_c_classes, relation_f_follow


def RelationCoarseArray_follow(num_c_classes, relation_follow):
    num_f_classes = len(relation_follow)
    # 2D: 0index, c; 1index, f
    relation_c_array_follow = np.full((num_f_classes + num_c_classes, num_f_classes), -1)

    for c in range(num_f_classes, num_f_classes + num_c_classes):
        num_perc = 0
        for f in range(num_f_classes):
            if c == relation_follow[f]:
                relation_c_array_follow[c, num_perc] = f
                num_perc += 1

    return relation_c_array_follow


def RelationCoarse_Single(c_label_single, relation_c_array):
    # Input: a coarse and relation_c_array
    # Output: a list of fine belong to the coarse

    flag = relation_c_array.shape[1]

    for f in range(relation_c_array.shape[1]):
        if relation_c_array[c_label_single][f] == -1:
            flag = f  # 记录第几列开始为-1
            break
    relation_c_single = relation_c_array[c_label_single, : flag]

    return relation_c_single


# ==================================================================
def FineToCoarse(f_labels, relation_f):
    # Input: a list of fine and relation_f
    # Output: a list of coarse

    c_labels = []
    for i in range(len(f_labels)):
        if f_labels[i] == -1:
            cl = -1
        else:
            cl = relation_f[int(f_labels[i])]
        c_labels.append(cl)

    return np.array(c_labels)


def FineToCoarseMiss(c_labels, num_c_classes):
    c_labels_entire = [i for i in range(num_c_classes)]
    c_labels_miss = [-1 if c_labels_entire[i] not in c_labels else c_labels_entire[i] for i in range(len(c_labels_entire))]

    return c_labels_miss


def HeadNeutralTail(num_f_classes, num_c_classes=None, relation_f=None):
    entire_f_marks = [i for i in range(num_f_classes)]
    head_f_marks = [i for i in entire_f_marks if i < num_f_classes // 3]
    tail_f_marks = [i for i in entire_f_marks if i >= num_f_classes - num_f_classes // 3]
    neutral_f_marks = [i for i in entire_f_marks if i not in head_f_marks and i not in tail_f_marks]

    if num_c_classes:
    # the list of labels from fine to coarse whose value is discontinuous,
    # i.e., miss some coarse, replace those with -1
        head_c_marks = FineToCoarse(head_f_marks, relation_f)
        tail_c_marks = FineToCoarse(tail_f_marks, relation_f)
        neutral_c_marks = FineToCoarse(neutral_f_marks, relation_f)

        head_c_marks, b = np.unique(head_c_marks, return_counts=True)  # [ 0  1  3  4  ...]
        tail_c_marks, b = np.unique(tail_c_marks, return_counts=True)
        neutral_c_marks, b = np.unique(neutral_c_marks, return_counts=True)

        head_c_marks = FineToCoarseMiss(head_c_marks, num_c_classes)  # [0, 1, -1, 3, 4, ...]
        tail_c_marks = FineToCoarseMiss(tail_c_marks, num_c_classes)
        neutral_c_marks = FineToCoarseMiss(neutral_c_marks, num_c_classes)

        return head_f_marks, neutral_f_marks, tail_f_marks, head_c_marks, neutral_c_marks, tail_c_marks
    return head_f_marks, neutral_f_marks, tail_f_marks


def HeadTail(num_cls, percls, tail_ratio, num_cls_c=None, relation=None):
    if tail_ratio > 1:  # depend on number. tail: less than 100 images.
        tail_marks = [i for i in range(len(percls)) if percls[i] <= tail_ratio]
    else:  # depend on ratio. 80%(0.8) tail.
        tail_marks = [i for i in range(num_cls) if i >= num_cls * (1 - tail_ratio)]
    head_marks = [i for i in range(num_cls) if i not in tail_marks]

    if num_cls_c:
        # the list of labels from fine to coarse whose value is discontinuous,
        # i.e., miss some coarse, replace those with -1
        head_c_marks = FineToCoarse(head_marks, relation)
        tail_c_marks = FineToCoarse(tail_marks, relation)

        head_c_marks, b = np.unique(head_c_marks, return_counts=True)  # [ 0  1  3  4  ...]
        tail_c_marks, b = np.unique(tail_c_marks, return_counts=True)

        head_c_marks = FineToCoarseMiss(head_c_marks, num_cls_c)  # [0, 1, -1, 3, 4, ...]
        tail_c_marks = FineToCoarseMiss(tail_c_marks, num_cls_c)

        return head_marks, tail_marks, head_c_marks, tail_c_marks

    return head_marks, tail_marks


def Relation_hier_tail(f_labels, hf_labels, tf_labels, relation):
    c_labels = []
    for i in range(len(f_labels)):
        if f_labels[i] in tf_labels:
            c_labels.append(relation[f_labels[i]] + len(hf_labels))
        else:
            c_labels.append(f_labels[i])

    relation = c_labels
    # ensure relation_f continuity
    a, b = np.unique(np.array(relation), return_counts=True)
    if a[len(a) - 1] != len(a) - 1:  # isMiss.  isMiss: [0, 1, 3, 4, 5] -> noMiss: [0, 1, 2, 3, 4]
        miss = []
        prior = -1
        for i in a:
            if i == 0:
                prior = i
                continue

            if i - prior == 1:
                prior = i
            else:
                dif = i - prior - 1
                for i_dif in range(dif):
                    miss.append(prior + 1 + i_dif)
                prior = i
        miss.sort(reverse=True)

        for i in miss:
            for j in range(len(relation)):
                if i < relation[j]:
                    relation[j] = relation[j] - 1

    return relation


# ==================================================================
def FineSort_Coarse(train_labels, train_labels_c):
    _, num_c_classes, relation_f = RelationFine1(train_labels, train_labels_c)
    relation_c_array = RelationCoarseArray(num_c_classes, relation_f)

    f_labels = []
    for i in range(num_c_classes):
        relation_c_single = RelationCoarse_Single(i, relation_c_array)
        relation_c_single = relation_c_single.tolist()
        f_labels += relation_c_single
    # print(f_labels)  # [4, 30, 55, 72, 95, 1, 32, ..., 85, 89]  depend on coarse

    f_labels_sort = list(range(len(f_labels)))  # []0, 1, 2, ..., 98, 99]

    b = np.array(f_labels)
    a = []
    for i in range(len(f_labels)):
        a.append(f_labels[i] - f_labels_sort[i])
    # print(a)  # [4, 29, 53, 69, 91, -4, 26, ..., -13, -10]

    return a, b


def Sort_HeadTail(train_labels, switch_imbalance):
    _, num_perclass_train, _, _ = dataset0.NumPerclass(train_labels, '', switch_imbalance)
    num_perclass = num_perclass_train
    # print(num_perclass)

    index_max_sort = []
    for i in num_perclass:
        index_max = np.where(np.array(num_perclass) == max(num_perclass))[0][0]
        index_max_sort.append(index_max)
        num_perclass[index_max] = -1
    # print(index_max_sort)  # [6, 15, 2, ..., 4, 23]

    f_labels = index_max_sort
    f_labels_sort = list(range(len(num_perclass)))  # [0, 1, 2, ..., 25, 26]

    b = np.array(f_labels)
    a = []
    for i in range(len(f_labels)):
        a.append(f_labels[i] - f_labels_sort[i])
    # print(a)  # [-6, -14, 0, ..., 21, 3]

    return a, b


# if __name__ == '__main__':


