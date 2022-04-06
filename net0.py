import net1, net2, net3


def ChooseNet(switch_net, device, num_cls, num_cls_c_semantic, num_cls_c_cluster, LDAM_net):
    if switch_net == 'ResNet10':
        model = net2.resnet10_hier(num_cls, num_cls_c_semantic, LDAM_net).to(device)
        model = net2.resnet10().to(device)
        model = net3.ResNet50(num_cls, num_cls_c_semantic).to(device)
    elif switch_net == 'ResNet18':
        model = resnet18
    elif switch_net == 'ResNet32':
        model = net1.resnet32_hier(num_cls, num_cls_c_semantic, LDAM_net).to(device)
    elif switch_net == 'ResNet32_2c':  # num_cls_c_semantic != num_cls_c_cluster
        model = net1.resnet32_hier_2c(num_cls, num_cls_c_semantic, num_cls_c_cluster, LDAM_net).to(device)
    elif switch_net == 'ResNet50':
        model = resnet50
    elif switch_net == 'ResNet101':
        model = resnet101
    elif switch_net == 'ResNet152':
        model = resnet152
    else:
        raise ValueError(switch_net)

    return model