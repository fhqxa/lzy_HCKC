import numpy as np, torch, torch.nn as nn

import comparison0


param_CB_c, param_CB_f, param_Focal = 0.9999, 0.9999, 1.0


def Criterion(num_cls, percls, switch_reweight, defer_epoch, switch_criterion, epoch, device):
    if switch_reweight == 'RW' and epoch == defer_epoch:
        percls_weight = 1 / np.array(percls)
        percls_weight = torch.FloatTensor(percls_weight).to(device)
    elif switch_reweight == 'CB' and epoch == defer_epoch:
        effective_num = 1.0 - np.power(param_CB_f, percls)
        percls_weight = (1.0 - param_CB_f) / np.array(effective_num)
        percls_weight = percls_weight / np.sum(percls_weight) * len(percls)
        percls_weight = torch.FloatTensor(percls_weight).to(device)
    else:
        percls_weight = torch.ones(num_cls).to(device)

    if switch_criterion == 'CE':
        criterion = nn.CrossEntropyLoss(weight=percls_weight).to(device)
    elif switch_criterion == 'Focal':
        criterion = comparison0.FocalLoss(percls_weight, param_Focal).to(device)
    elif switch_criterion == 'LDAM':
        criterion = comparison0.LDAMLoss(device, percls, weight=percls_weight).to(device)
    else:
        raise ValueError(switch_criterion)

    return criterion


def Criterion_c(num_cls_c_semantic, num_cls_c_cluster, percls_c_semantic, percls_c_cluster,
                switch_reweight, defer_epoch, switch_criterion, epoch, device):
    if switch_reweight == 'RW' and epoch == defer_epoch:
        percls_weight_c_semantic = 1 / np.array(percls_c_semantic)
        percls_weight_c_cluster = 1 / np.array(percls_c_cluster)

        percls_weight_c_semantic = torch.FloatTensor(percls_weight_c_semantic).to(device)
        percls_weight_c_cluster = torch.FloatTensor(percls_weight_c_cluster).to(device)
    elif switch_reweight == 'CB' and epoch == defer_epoch:
        effective_num_c = 1.0 - np.power(param_CB_c, percls_c_semantic)
        percls_weight_c_semantic = (1.0 - param_CB_c) / np.array(effective_num_c)
        percls_weight_c_semantic = percls_weight_c_semantic / np.sum(percls_weight_c_semantic) * len(percls_c_semantic)

        effective_num_c = 1.0 - np.power(param_CB_c, percls_c_cluster)
        percls_weight_c_cluster = (1.0 - param_CB_c) / np.array(effective_num_c)
        percls_weight_c_cluster = percls_weight_c_cluster / np.sum(percls_weight_c_cluster) * len(percls_c_cluster)

        percls_weight_c_semantic = torch.FloatTensor(percls_weight_c_semantic).to(device)
        percls_weight_c_cluster = torch.FloatTensor(percls_weight_c_cluster).to(device)
    else:
        percls_weight_c_semantic = torch.ones(num_cls_c_semantic).to(device)
        percls_weight_c_cluster = torch.ones(num_cls_c_cluster).to(device)

    if switch_criterion == 'CE':
        criterion_c_semantic = nn.CrossEntropyLoss(weight=percls_weight_c_semantic).to(device)
        criterion_c_cluster = nn.CrossEntropyLoss(weight=percls_weight_c_cluster).to(device)
    elif switch_criterion == 'Focal':
        criterion_c_semantic = comparison0.FocalLoss(percls_weight_c_semantic, param_Focal).to(device)
        criterion_c_cluster = comparison0.FocalLoss(percls_weight_c_cluster, param_Focal).to(device)
    elif switch_criterion == 'LDAM':
        criterion_c_semantic = comparison0.LDAMLoss(device, percls_c_semantic, percls_weight_c_semantic).to(device)
        criterion_c_cluster = comparison0.LDAMLoss(device, percls_c_cluster, percls_weight_c_cluster).to(device)
    else:
        raise ValueError(switch_criterion)

    return criterion_c_semantic, criterion_c_cluster
