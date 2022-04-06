run_i = [0, 1, 2]
run_j = [0, 1, 3, 4, 5, 7, 8]  # switch_rebalance3
run_j = ['_5', '_10', '_15', '_20', '_25', '_30']  # cluster_num
run_j = [0, 7]

run_i = [0]
run_j = [0]
print(run_i, run_j)
for i_run in run_i:
    for j_run in run_j:
        log_mark = '_cluster_num'
        sw_mark = j_run

        log_mark = ''
        sw_mark = ''
        print('log_mark:', log_mark, 'sw_mark:', sw_mark)

        if len(run_i) > 1 or len(run_j) > 1:
            print('run plan: {} - {}'.format(i_run, j_run))

        # ==========================================================================================
        CUDA = 1

        tail_ratio           = 0  # flat-0, 100%-1, 90%-2, 80%-3, 70%-4, 60%-5, 100-6
        hier_loss            = 1
        transfer_epoch       = 5 if tail_ratio != 0 else 'x'
        cluster_epoch        = 2 if tail_ratio != 0 else 'x'
        net_coarse_layer     = 0 if tail_ratio != 0 else 'x'
        cluster_coarse_layer = 0 if tail_ratio != 0 else 'x'

        switch_criterion3    = 2  # 'CE'-0, 'Focal'-1, 'LDAM'-2
        switch_rebalance3    = 7  # 'None'-0, 'RS'-1, 'SMOTE'-2, 'RW'-3, 'CB'-4, 'DRS'-5, 'DSMOTE'-6, 'DRW'-7, 'DCB'-8
        switch_optimizer3    = 0  # 'SGD'-0, 'Adam'-1, 'Adagrad'-2,
        switch_lr3           = 0
        switch_net           = 3  # 'ResNet10'-0, 'ResNet18'-1, 'ResNet32'-2, 'ResNet32_2c'-3, 'ResNet50'-4,
        switch_dataset       = 5  # 'CIFAR-10',        'CIFAR-100',       'tieredImageNet',  'ImageNet2012',
                                  # 'VOC2007_Per'-4,   'VOC2012_Per',     'VOC2007_PerBir',  'VOC2012_PerBir'
                                  # 'SUN397'-8,        'iNaturalist2017',
        switch_imbalance     = 3  # 'Original'-0, 'IR10'-1, 'IR20'-2, 'IR50'-3, 'IR100'-4, 'IR200'-5
        coarse_epoch, hier_epoch, fine_epoch = (55, 45, 200) if tail_ratio != 0 else (0, 0, 300)
        # coarse_epoch, hier_epoch, fine_epoch = (60, 60, 120) if tail_ratio != 0 else (0, 0, 240)

        # switch_criterion3 = i_run
        # switch_rebalance3 = j_run
        # cluster_epoch = j_run

        # ==========================================================================================
        isTest = False

        def Switch():
            global tail_ratio, hier_loss, transfer_epoch, cluster_epoch, net_coarse_layer, cluster_coarse_layer, \
                switch_criterion3, switch_rebalance3, switch_optimizer3, switch_lr3, \
                switch_net, switch_dataset, switch_imbalance

            switch = str(tail_ratio) + str(hier_loss) + str(transfer_epoch) + \
                     str(cluster_epoch) + str(net_coarse_layer) + str(cluster_coarse_layer) + '_' + \
                     str(switch_criterion3) + str(switch_rebalance3) + str(switch_optimizer3) + str(switch_lr3) + \
                     str(switch_net) + str(switch_dataset) + str(switch_imbalance)

            def f1(flag=''):  # index of sw
                global i_sw_f1

                if flag == '':
                    try:
                        i_sw_f1 += 1
                    except:
                        i_sw_f1 = 0

                    if switch[i_sw_f1] not in '0123456789':  #
                        i_sw_f1 += 1
                    return int(switch[i_sw_f1])  # 0~9
                else:
                    if flag == 'x':
                        try:
                            i_sw_f1 += 1
                        except:
                            i_sw_f1 = 0

                        return False
                    else:
                        return True

            tail_ratio              = [0,                  1,                  0.9,                0.8,
                                       0.7,                0.6,                100,                '',                ][f1()]
            hier_loss               = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][f1()]
            transfer_epoch          = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][f1()] if f1(transfer_epoch) else 'x'
            cluster_epoch           = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][f1()] if f1(cluster_epoch) else 'x'
            net_coarse_layer        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][f1()] if f1(net_coarse_layer) else 'x'
            cluster_coarse_layer    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][f1()] if f1(cluster_coarse_layer) else 'x'

            switch_criterion3       = ['CE',               'Focal',            'LDAM',                                ][f1()]
            switch_rebalance3       = ['None',             'RS',               'SMOTE',            'RW',
                                       'CB',               'DRS',              'DSMOTE',           'DRW',
                                       'DCB',                                                                         ][f1()]
            switch_optimizer3       = ['SGD',              'Adam',             'Adagrad', ][f1()]
            switch_lr3              = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9                                                   ][f1()]
            switch_net              = ['ResNet10',         'ResNet18',         'ResNet32',         'ResNet32_2c',
                                       'ResNet50',         'ResNet101',        'ResNet152',                           ][f1()]
            switch_dataset          = ['CIFAR-10',         'CIFAR-100',        'tieredImageNet',   'ImageNet2012',
                                       'VOC2007_Per',      'VOC2012_Per',      'VOC2007_PerBir',   'VOC2012_PerBir',
                                       'SUN397',           'iNaturalist2017'                                          ][f1()]
            switch_imbalance        = ['Original',         'IR10',             'IR20',             'IR50',
                                       'IR100',            'IR200',                                                   ][f1()]

            global cluster_num
            if switch_dataset == 'CIFAR-10':
                cluster_num = 1
            elif switch_dataset == 'CIFAR-100':
                cluster_num = 10  # 1;'equal_semantic'
            elif switch_dataset == 'tieredImageNet':
                cluster_num = 10
            elif switch_dataset == 'ImageNet2012':
                cluster_num = 1
            elif switch_dataset == 'VOC2012_Per':
                switch_imbalance = 'Original'
                cluster_num = 6
            elif switch_dataset == 'VOC2012_PerBir':
                switch_imbalance = 'Original'
                cluster_num = 4
            elif switch_dataset == 'SUN397':
                switch_imbalance = 'Original'
                cluster_num = 5
            elif switch_dataset == 'iNaturalist2017':
                switch_imbalance = 'Original'
                cluster_num = 5

            return switch
        switch = Switch()
        i_sw_f1 = -1

        end_epoch = coarse_epoch + hier_epoch + fine_epoch
        import os, torch, numpy as np
        SEED = 0  # 0;None
        SEED = SEED if SEED is not None else np.random.randint(10000)
        device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')

        directory_log0 = f'{os.getcwd()}/log{log_mark}/{switch_dataset}_{switch_imbalance}_{switch_net}_epoch{end_epoch}' \
                         f'_SEED{SEED}/'.replace('/all_code/', '/all_log/')
        if tail_ratio == 0:
            directory_log = directory_log0 + f'flat_epoch_c{coarse_epoch}h{hier_epoch}f{fine_epoch}/sw{switch}/'
        else:
            directory_log = directory_log0 + f'tail_ratio{tail_ratio}_epoch_c{coarse_epoch}h{hier_epoch}f{fine_epoch}' \
                                             f'/sw{switch}{sw_mark}/'

        system = 0  # require to change the path, when move the entire project among different systmes
        if system == 0:
            path_all_dataset = '/home/lzy/Datasets/'
            path_all_dataset_usb = '/media/lzy/大U盘/Datasets/'
            path_custom_tool = '/home/lzy/PycharmProjects/pythonProject/1/'
        elif system == 1:
            path_all_dataset = ''
            path_custom_tool = ''

        # ==========================================================================================
        import numpy as np, torch, torch.nn as nn, os
        from tqdm import tqdm
        from datetime import datetime
        from sklearn.metrics import confusion_matrix

        import comparison0, criterion0, dataset0, label0, method0, net0, optimizer0, utility0

        random_name = utility0.RandomName()
        utility0.FixSeed(SEED)

        image_path, image_path_test, transform, transform_test, num_cls, num_cls_c_semantic, relation_semantic, \
        eval_tree = \
            dataset0.AllDatasets(switch_dataset, path_all_dataset, path_all_dataset_usb, switch_imbalance)

        train_dataset = dataset0.DatasetFromPath(image_path, transform)
        test_dataset = dataset0.DatasetFromPath(image_path_test, transform_test)
        train_labels, test_labels = [i[1] for i in image_path], [i[1] for i in image_path_test]
        _, percls, percls_test, _ = dataset0.NumPerclass(train_labels, test_labels, switch_imbalance)

        head_marks2, tail_marks2 = label0.HeadTail(num_cls, percls, tail_ratio)
        if tail_ratio != 0:  # semantic and cluster
            relation_semantic = label0.Relation_hier_tail(
                [i for i in range(num_cls)], head_marks2, tail_marks2, relation_semantic)
            a, _ = np.unique(np.array(relation_semantic), return_counts=True)
            num_cls_c_semantic = len(a)

            if cluster_num == 'equal_semantic':
                num_cls_c_cluster = num_cls_c_semantic
                cluster_num = num_cls_c_cluster - len(head_marks2)
            else:
                num_cls_c_cluster = len(head_marks2) + cluster_num

            if '_cluster_num' in log_mark:
                cluster_num = int(sw_mark.split('_')[1])
                num_cls_c_cluster = len(head_marks2) + cluster_num
        else:
            cluster_num = 1
            num_cls_c_cluster = num_cls_c_semantic

        percls_c_semantic = dataset0.NumPerclass_Coarse(num_cls, num_cls_c_semantic, relation_semantic, percls)

        LDAM_net = True if switch_criterion3 == 'LDAM' else False
        model = net0.ChooseNet(switch_net, device, num_cls, num_cls_c_semantic, num_cls_c_cluster, LDAM_net)
        if isTest:
            directory_log_test, random_name = utility0.LoadModel_test(model, directory_log)
        else:
            directory_txt, directory_csv, csv_header, directory_csv_cf_matrix = \
                utility0.CreateLog(random_name, directory_log, isTest)


        def Train(train_test, model, data_loader):  # train_test: 1-train; 0-test
            sum_num = sum_l0 = sum_l = sum_l_c = sum_acc = sum_acc_c = 0

            if train_test:
                feature_cluster = label_cluster = ''
                model.train()
            else:
                sum_acc5, all_label, all_pred = 0, [], []
                model.eval()

            for i, (data, target) in enumerate(data_loader):
                batch_size = target.size(0)
                sum_num += batch_size

                data = data.to(device)
                out, out_c, feature = model(data)

                if train_test:
                    feature_cluster, label_cluster = method0.ClusterEpoch(
                        True, epoch, relation_semantic, relation_cluster, feature_cluster, label_cluster, feature, target,
                        tail_marks2, model, train_loader, cluster_num, cluster_coarse_layer, device,
                        cluster_epoch, coarse_epoch, hier_epoch)

                label = target.long().to(device)

                label_c = label0.FineToCoarse(label, relation)
                label_c = torch.from_numpy(label_c).long().to(device)

                if param_cluster_epoch:
                    criterion_c = criterion_c_cluster
                    if '_2c' in switch_net:
                        out_c = out_c[1]
                else:
                    criterion_c = criterion_c_semantic
                    if '_2c' in switch_net:
                        out_c = out_c[0]

                param_transfer_epoch = method0.TransferEpoch(
                    epoch, transfer_epoch, coarse_epoch, hier_epoch, fine_epoch, end_epoch,
                    switch_imbalance)

                if train_test:
                    l0, l_c, l = method0.HierLoss(
                        criterion, criterion_c, out, out_c, label, label_c, param_transfer_epoch)
                else:
                    l0, l_c, l = method0.HierLoss(
                        criterion_test, criterion_test, out, out_c, label, label_c, param_transfer_epoch)

                sum_l0 += l0.item() * batch_size
                sum_l += l.item() * batch_size
                sum_l_c += l_c.item() * batch_size

                topk = (1,) if train_test else (1, 5)
                _, pred = out.topk(max(topk), 1, True, True)  # topk(k, dim, largest, sorted)
                correct = pred.eq(label.view(-1, 1).expand_as(pred))
                acc_topk = []
                for k in topk:
                    correct_k = correct[:, : k].reshape(-1).float().sum()
                    acc_topk.append(correct_k)
                sum_acc += acc_topk[0].item()

                topk_c = (1,)
                _, pred_c = out_c.topk(max(topk_c), 1, True, True)
                correct_c = pred_c.eq(label_c.view(-1, 1).expand_as(pred_c))
                acc_c_topk = []
                for k in topk_c:
                    correct_c_k = correct_c[:, : k].reshape(-1).float().sum()
                    acc_c_topk.append(correct_c_k)
                sum_acc_c += acc_c_topk[0].item()

                if train_test:
                    # Back propagate
                    optimizer.zero_grad()
                    l0.backward()
                    optimizer.step()
                else:
                    sum_acc5 += acc_topk[1].item()
                    all_label.extend(label.cpu().numpy().flatten())
                    all_pred.extend(pred[:, : 1].cpu().numpy().flatten())

            if train_test:
                reserve = {'l_train': sum_l0 / sum_num, 'l_c_train': sum_l_c / sum_num, 'l_f_train': sum_l / sum_num,
                           'acc_c_train': sum_acc_c / sum_num, 'acc_f_train': sum_acc / sum_num,
                           'learn_rate': param_lr, 'transfer_epoch': param_transfer_epoch,
                           'cluster_epoch': param_cluster_epoch}
                return reserve, feature_cluster, label_cluster
            else:
                # np.set_printoptions(threshold=np.inf)  # no default print
                cf_matrix = confusion_matrix(all_label, all_pred)
                cls_hit_tie = utility0.EvaHier_TreeInducedError(eval_tree, all_pred, all_label, switch_dataset)
                cls_hit_fh = utility0.EvaHier_HierarchicalPrecisionAndRecall(eval_tree, all_pred, all_label)

                reserve = {'l': sum_l0 / sum_num, 'l_c': sum_l_c / sum_num, 'l_f': sum_l / sum_num,
                           'acc_c': sum_acc_c / sum_num, 'acc_f': sum_acc / sum_num, 'acc5_f': sum_acc5 / sum_num,
                           'fh_f': sum(cls_hit_fh) / sum_num, 'tie_f': sum(cls_hit_tie) / sum_num}

                return reserve, (cf_matrix, cls_hit_fh, cls_hit_tie)


        def Train1(train_test, model, data_loader):  # train_test: 1-train; 0-test
            sum_num = sum_l0 = sum_l = sum_l_c = sum_acc = sum_acc_c = 0

            if train_test:
                feature_cluster = label_cluster = ''
                model.train()
            else:
                sum_acc5, all_label, all_pred = 0, [], []
                model.eval()

            for i, (data, target) in enumerate(data_loader):
                batch_size = target.size(0)
                sum_num += batch_size

                data = data.to(device)
                out, out_c, feature = model(data)

                if train_test:
                    feature_cluster, label_cluster = method0.ClusterEpoch(
                        True, epoch, relation_semantic, relation_cluster, feature_cluster, label_cluster, feature, target,
                        tail_marks2, model, train_loader, cluster_num, cluster_coarse_layer, device,
                        cluster_epoch, coarse_epoch, hier_epoch)

                label = target.long().to(device)
                label_c = label0.FineToCoarse(label, relation)
                label_c = torch.from_numpy(label_c).long().to(device)

                label_c_semantic = label0.FineToCoarse(label, relation_semantic)
                label_c_cluster = label0.FineToCoarse(label, relation_cluster)
                label_c_semantic = torch.from_numpy(label_c_semantic).long().to(device)
                label_c_cluster = torch.from_numpy(label_c_cluster).long().to(device)

                param_transfer_epoch = method0.TransferEpoch(
                    epoch, transfer_epoch, coarse_epoch, hier_epoch, fine_epoch, end_epoch,
                    switch_imbalance)

                param_hier_loss1 = method0.HierLoss1_epoch(
                    hier_loss, epoch, coarse_epoch, hier_epoch, fine_epoch, end_epoch)

                if train_test:
                    l0, l_c, l = method0.HierLoss1(
                        criterion, criterion_c_semantic, criterion_c_cluster, out, out_c,
                        label, label_c_semantic, label_c_cluster,
                        param_hier_loss1, param_transfer_epoch)
                else:
                    l0, l_c, l = method0.HierLoss1(
                        criterion_test, criterion_test, criterion_test, out, out_c,
                        label, label_c_semantic, label_c_cluster,
                        param_hier_loss1, param_transfer_epoch)

                sum_l0 += l0.item() * batch_size
                sum_l += l.item() * batch_size
                sum_l_c += l_c.item() * batch_size

                topk = (1,) if train_test else (1, 5)
                _, pred = out.topk(max(topk), 1, True, True)  # topk(k, dim, largest, sorted)
                correct = pred.eq(label.view(-1, 1).expand_as(pred))
                acc_topk = []
                for k in topk:
                    correct_k = correct[:, : k].reshape(-1).float().sum()
                    acc_topk.append(correct_k)
                sum_acc += acc_topk[0].item()

                topk_c = (1,)
                _, pred_c = out_c.topk(max(topk_c), 1, True, True)
                correct_c = pred_c.eq(label_c.view(-1, 1).expand_as(pred_c))
                acc_c_topk = []
                for k in topk_c:
                    correct_c_k = correct_c[:, : k].reshape(-1).float().sum()
                    acc_c_topk.append(correct_c_k)
                sum_acc_c += acc_c_topk[0].item()

                if train_test:
                    # Back propagate
                    optimizer.zero_grad()
                    l0.backward()
                    optimizer.step()
                else:
                    sum_acc5 += acc_topk[1].item()
                    all_label.extend(label.cpu().numpy().flatten())
                    all_pred.extend(pred[:, : 1].cpu().numpy().flatten())

            if train_test:
                reserve = {'l_train': sum_l0 / sum_num, 'l_c_train': sum_l_c / sum_num, 'l_f_train': sum_l / sum_num,
                           'acc_c_train': sum_acc_c / sum_num, 'acc_f_train': sum_acc / sum_num,
                           'learn_rate': param_lr, 'transfer_epoch': param_transfer_epoch,
                           'cluster_epoch': param_cluster_epoch}
                return reserve, feature_cluster, label_cluster
            else:
                # np.set_printoptions(threshold=np.inf)  # no default print
                cf_matrix = confusion_matrix(all_label, all_pred)
                cls_hit_tie = utility0.EvaHier_TreeInducedError(eval_tree, all_pred, all_label, switch_dataset)
                cls_hit_fh = utility0.EvaHier_HierarchicalPrecisionAndRecall(eval_tree, all_pred, all_label)

                reserve = {'l': sum_l0 / sum_num, 'l_c': sum_l_c / sum_num, 'l_f': sum_l / sum_num,
                           'acc_c': sum_acc_c / sum_num, 'acc_f': sum_acc / sum_num, 'acc5_f': sum_acc5 / sum_num,
                           'fh_f': sum(cls_hit_fh) / sum_num, 'tie_f': sum(cls_hit_tie) / sum_num}

                return reserve, (cf_matrix, cls_hit_fh, cls_hit_tie)


        criterion_test = nn.CrossEntropyLoss()
        flag_c = flag_h = flag_f = 1
        best_result = {'best_acc_c': [0, 0], 'best_acc_f': [0, 0], 'best_acc5_f': [0, 0], 'best_fh_f': [0, 0],
                       'best_tie_f': [1, 0],
                       'best_cf_matrix': ''}  # best_acc_c = [a, b]: a-acc; b-epoch
        relation_cluster = feature_cluster = label_cluster = feature = target = 'none'

        print('\nCUDA: {}\n{}\nsw{}\n'.format(CUDA, directory_log0.split(f'/all_log/', 1)[1], switch))

        progress_bar = tqdm(range(end_epoch)) if not isTest else tqdm(range(1))
        for epoch in progress_bar:
            if True:
                transfer_stage = 0  # stage: coarse-1, hier-2, fine-3
                if transfer_stage == 0:
                    switch_criterion1 = switch_criterion2 = switch_criterion3
                    switch_rebalance1 = switch_rebalance2 = 'None'
                    switch_optimizer1 = switch_optimizer2 = switch_optimizer3
                    switch_lr1 = switch_lr2 = 5
                    defer_epoch1, defer_epoch2, defer_epoch3 = 4 / 5 * coarse_epoch, coarse_epoch + 4 / 5 * hier_epoch, \
                                                               coarse_epoch + hier_epoch + 4 / 5 * fine_epoch
                    batch_size1 = batch_size2 = batch_size3 = 128
                if epoch < coarse_epoch and coarse_epoch > 0:
                    if flag_c:
                        flag_c = 0

                        switch_criterion = switch_criterion1
                        switch_defer, switch_resample, switch_reweight = comparison0.Rebalance(switch_rebalance1)
                        switch_optimizer = switch_optimizer1
                        switch_lr = switch_lr1
                        defer_epoch = defer_epoch1 if switch_defer else 0
                        batch_size = batch_size1
                        train_loader, test_loader, _ = dataset0.DataloaderFromDataset(
                            train_dataset, test_dataset, percls, percls_test, switch_resample,
                            defer_epoch, batch_size, epoch)
                        criterion = criterion0.Criterion(
                            num_cls, percls, switch_reweight, defer_epoch, switch_criterion, epoch, device)
                        optimizer, model = optimizer0.Optimizer(switch_optimizer, model)
                elif coarse_epoch <= epoch < coarse_epoch + hier_epoch and hier_epoch > 0:
                    if flag_h:
                        flag_h = 0

                        switch_criterion = switch_criterion2
                        switch_defer, switch_resample, switch_reweight = comparison0.Rebalance(switch_rebalance2)
                        switch_optimizer = switch_optimizer2
                        switch_lr = switch_lr2
                        defer_epoch = defer_epoch2 if switch_defer else coarse_epoch
                        batch_size = batch_size2
                        train_loader, test_loader, _ = dataset0.DataloaderFromDataset(
                            train_dataset, test_dataset, percls, percls_test, switch_resample,
                            defer_epoch, batch_size, epoch)
                        criterion = criterion0.Criterion(
                            num_cls, percls, switch_reweight, defer_epoch, switch_criterion, epoch, device)
                        optimizer, model = optimizer0.Optimizer(switch_optimizer, model)
                else:
                    if flag_f:
                        flag_f = 0

                        switch_criterion = switch_criterion3
                        switch_defer, switch_resample, switch_reweight = comparison0.Rebalance(switch_rebalance3)
                        switch_optimizer = switch_optimizer3
                        switch_lr = switch_lr3
                        defer_epoch = defer_epoch3 if switch_defer else coarse_epoch + hier_epoch
                        batch_size = batch_size3
                        train_loader, test_loader, _ = dataset0.DataloaderFromDataset(
                            train_dataset, test_dataset, percls, percls_test, switch_resample,
                            defer_epoch, batch_size, epoch)
                        criterion = criterion0.Criterion(
                            num_cls, percls, switch_reweight, defer_epoch, switch_criterion, epoch, device)
                        optimizer, model = optimizer0.Optimizer(switch_optimizer, model)

            if switch_defer and epoch == defer_epoch:
                train_loader, test_loader, _ = dataset0.DataloaderFromDataset(
                    train_dataset, test_dataset, percls, percls_test, switch_resample,
                    defer_epoch, batch_size, epoch)
                criterion = criterion0.Criterion(
                    num_cls, percls, switch_reweight, defer_epoch, switch_criterion, epoch, device)

            relation, relation_semantic, relation_cluster, param_cluster_epoch = method0.ClusterEpoch(
                False, epoch, relation_semantic, relation_cluster, feature_cluster, label_cluster, feature, target,
                tail_marks2, model, train_loader, cluster_num, cluster_coarse_layer, device,
                cluster_epoch, coarse_epoch, hier_epoch)

            if param_cluster_epoch:
                percls_c_cluster = dataset0.NumPerclass_Coarse(num_cls, num_cls_c_cluster, relation_cluster, percls)
            else:
                percls_c_cluster = percls_c_semantic
            criterion_c_semantic, criterion_c_cluster = criterion0.Criterion_c(
                num_cls_c_semantic, num_cls_c_cluster, percls_c_semantic, percls_c_cluster,
                switch_reweight, defer_epoch, switch_criterion, epoch, device)

            param_lr = utility0.AdjustLearnRate(epoch, optimizer, switch_lr, coarse_epoch, hier_epoch, fine_epoch)

            if not isTest:
                reserve_train, feature_cluster, label_cluster = Train(1, model, train_loader)
            reserve, cf_matrix = Train(0, model, test_loader)
            # reserve_train, feature_cluster, label_cluster = Train1(1, model, train_loader, criterion, criterion_c_semantic, criterion_c_cluster) if not isTest else ''
            # reserve, cf_matrix = Train1(0, model, test_loader)

            if not isTest:
                reserve.update(reserve_train)  # reserve.update({'learn_rate': value_lr})
                utility0.SaveCsv(reserve, directory_csv, csv_header)
                best_result = utility0.SaveBestResult(reserve, cf_matrix, model, epoch, best_result, switch, directory_log)
            else:
                flag = 0

                if flag == 0:
                    fea_all = ''
                    for data, target in test_loader:
                        data = data.to(device)
                        # fea, _, _ = model(data)
                        _, _, fea = model(data)
                        fea = fea[6]

                        fea = fea.cpu().detach().numpy()
                        target = target.cpu().numpy()

                        if isinstance(fea_all, str):
                            fea_all = fea
                            label_all = target
                        else:
                            # fea_all = torch.cat((fea_all, fea))
                            fea_all = np.vstack((fea_all, fea))
                            label_all = np.hstack((label_all, target))

                    fea_all = fea_all.reshape(fea_all.shape[0], -1)

                    utility0.Unique1(label_all)
                    label_select_idx = np.where(label_all < 2)[0]
                    label_select = label_all[label_select_idx]
                    utility0.Unique1(label_select)

                    fea_select = fea_all[label_select_idx]

                    utility0.FeatureVisual('t_SNE', fea_select, label_select, device, directory_log_test, random_name)
                    utility0.FeatureVisual('PCA', fea_select, label_select, device, directory_log_test, random_name)

            message = 'l: {:.4f}, acc_f: {:.4f}'.format(reserve['l'], reserve['acc_f'])
            progress_bar.set_description(message)
            # progress_bar.set_postfix(CUDA=CUDA)
            # if epoch < 5:
            #     print()

        # utility0.DrawOnCsv_loss('l', file_csv_log, directory_log, id_name_log)

        # =====================================================================================================
        if not isTest:
            utility0.SaveLog('best:\nacc_c: {:.4f}; acc_f: {:.4f}; acc5_f: {:.4f}; fh_f: {:.4f}; tie_f: {:.4f}\n'.format(
                best_result['best_acc_c'][0], best_result['best_acc_f'][0], best_result['best_acc5_f'][0],
                best_result['best_fh_f'][0], best_result['best_tie_f'][0]), directory_txt, True)

            cf_matrix, cls_hit_fh, cls_hit_tie = \
                best_result['best_cf_matrix'][0], best_result['best_cf_matrix'][1], best_result['best_cf_matrix'][2]
            np.savetxt(directory_csv_cf_matrix, cf_matrix, fmt='%.4f', delimiter=',')
            cls_hit_acc, cls_cnt = np.diag(cf_matrix), cf_matrix.sum(axis=1)
            if len(head_marks2) != 0 and len(tail_marks2) != 0:
                utility0.SaveLog('head/tail (acc): {:.2f}/{:.2f}\n'
                                 'head: {}\n{}\ntail: {}\n{}\n'.format(
                    sum(cls_hit_acc[: len(head_marks2)]) / len(head_marks2),
                    sum(cls_hit_acc[len(head_marks2):]) / len(tail_marks2),
                    len(head_marks2), head_marks2, len(tail_marks2), tail_marks2), directory_txt)
            utility0.SaveLog('cls_hit_acc: \n{}\ncls_hit_fh: \n{}\ncls_hit_tie: \n{}\ncls_cnt: \n{}\n'.format(
                cls_hit_acc, cls_hit_fh, cls_hit_tie, cls_cnt), directory_txt)

            utility0.SaveLog('num_cls_c_semantic: {}\nrelation_semantic: \n{}\n'
                             'num_cls_c_cluster: {}\nrelation_cluster: \n{}\n'.format(
                num_cls_c_semantic, np.array(relation_semantic),
                num_cls_c_cluster, np.array(relation_cluster)), directory_txt)

            utility0.SaveLog('num_perclass_test: \n{}\nnum_perclass_train_c: \n{}\n'.format(
                np.array(percls_test), np.array(percls_c_semantic)), directory_txt)

            utility0.SaveLog('fine_to_coarse: c_label(c_num) [f_label][[f_num]]', directory_txt)

            name_c_classes = [i for i in range(num_cls_c_semantic)]
            name_f_classes = [i for i in range(num_cls)]
            utility0.VisualRelation_fine_to_coarse(name_c_classes, percls_c_semantic, name_f_classes,
                                                   percls, relation_semantic, directory_txt)

            utility0.VisualLearningRate(directory_txt, directory_csv)

            utility0.SaveLog('directory_log: {}, \nimage_path: {}'.format(
                directory_log, image_path[0]), directory_txt)

            str_acc_f = utility0.AccToString(best_result['best_acc_f'][0])
            os.rename(directory_log, directory_log.rstrip('/') + f'_f{str_acc_f}/')
            # os.rename(directory_log, directory_log.rstrip('/') + f'_f{str_acc_f}_c{coarse_epoch}h{hier_epoch}f{fine_epoch}/')
