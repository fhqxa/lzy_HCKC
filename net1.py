# from Deep Residual Learning for Image Recognition.
# from LDAM - Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
# ResNet20, 32, 44, 56, 110, 1202


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_f_classes=10, LDAM_net=False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if LDAM_net:
            self.linear = NormedLinear(64, num_f_classes)
        else:
            self.linear = nn.Linear(64, num_f_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_s_hier(nn.Module):
    # 加载完数据（数据集）后，加载网络会导致数据的数值变化，原因不明。
    # 不同的网络构造会导致不同的数值变化，最终导致损失不同和最优精度偏差。
    # 网络构造: def __init__()
    # 不同的网络构造: self.linear 和 self.linear + self.linear_c
    def __init__(self, block, num_blocks, num_cls, num_cls_c, LDAM_net=False):
        super(ResNet_s_hier, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if LDAM_net:
            self.linear = NormedLinear(64, num_cls)
            self.linear_c = NormedLinear(64, num_cls_c)
        else:
            self.linear = nn.Linear(64, num_cls)
            self.linear_c = nn.Linear(64, num_cls_c)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)  # torch.Size([100, 3, 32, 32])

        out_conv1_1 = self.conv1(x)
        out_conv1_2 = self.bn1(out_conv1_1)
        out_conv1_3 = F.relu(out_conv1_2)
        # print(out_conv1_1.shape)  # torch.Size([100, 16, 32, 32])
        # print(out_conv1_1.shape)  # torch.Size([100, 16, 32, 32])
        # print(out_conv1_1.shape)  # torch.Size([100, 16, 32, 32])

        out_layer1 = self.layer1(out_conv1_3)
        out_layer2 = self.layer2(out_layer1)
        out_layer3 = self.layer3(out_layer2)
        # print(out_layer1.shape)  # torch.Size([100, 16, 32, 32])
        # print(out_layer2.shape)  # torch.Size([100, 32, 16, 16])
        # print(out_layer3.shape)  # torch.Size([100, 64, 8, 8])

        out_pool = F.avg_pool2d(out_layer3, out_layer3.size()[3])
        # print(out_pool.shape)  # torch.Size([100, 64, 1, 1])
        out_pool = out_pool.view(out_pool.size(0), -1)
        # print(out_pool.shape)  # torch.Size([100, 64])

        out_f = self.linear(out_pool)
        out_c = self.linear_c(out_pool)
        # print(out_f.shape)  # torch.Size([100, 100])
        # print(out_c.shape)  # torch.Size([100, 20])

        return out_f, out_c, \
               (out_pool, out_layer3, out_layer2, out_layer1, out_conv1_3, out_conv1_2, out_conv1_1)


class ResNet_s_hier_2c(nn.Module):
    # 加载完数据（数据集）后，加载网络会导致数据的数值变化，原因不明。
    # 不同的网络构造会导致不同的数值变化，最终导致损失不同和最优精度偏差。
    # 网络构造: def __init__()
    # 不同的网络构造: self.linear 和 self.linear + self.linear_c
    def __init__(self, block, num_blocks, num_cls, num_cls_c_semantic, num_cls_c_cluster, LDAM_net=False):
        super(ResNet_s_hier_2c, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if LDAM_net:
            self.linear = NormedLinear(64, num_cls)
            self.linear_c_semantic = NormedLinear(64, num_cls_c_semantic)
            self.linear_c_cluster = NormedLinear(64, num_cls_c_cluster)
        else:
            self.linear = nn.Linear(64, num_cls)
            self.linear_c_semantic = nn.Linear(64, num_cls_c_semantic)
            self.linear_c_cluster = nn.Linear(64, num_cls_c_cluster)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)  # torch.Size([100, 3, 32, 32])

        out_conv1_1 = self.conv1(x)
        out_conv1_2 = self.bn1(out_conv1_1)
        out_conv1_3 = F.relu(out_conv1_2)
        # print(out_conv1_1.shape)  # torch.Size([100, 16, 32, 32])
        # print(out_conv1_1.shape)  # torch.Size([100, 16, 32, 32])
        # print(out_conv1_1.shape)  # torch.Size([100, 16, 32, 32])

        out_layer1 = self.layer1(out_conv1_3)
        out_layer2 = self.layer2(out_layer1)
        out_layer3 = self.layer3(out_layer2)
        # print(out_layer1.shape)  # torch.Size([100, 16, 32, 32])
        # print(out_layer2.shape)  # torch.Size([100, 32, 16, 16])
        # print(out_layer3.shape)  # torch.Size([100, 64, 8, 8])

        out_pool = F.avg_pool2d(out_layer3, out_layer3.size()[3])
        # print(out_pool.shape)  # torch.Size([100, 64, 1, 1])
        out_pool = out_pool.view(out_pool.size(0), -1)
        # print(out_pool.shape)  # torch.Size([100, 64])

        out_f = self.linear(out_pool)
        out_c_semantic = self.linear_c_semantic(out_pool)
        out_c_cluster = self.linear_c_cluster(out_pool)
        # print(out_f.shape)  # torch.Size([100, 100])
        # print(out_c_semantic.shape)  # torch.Size([100, 20])
        # print(out_c_cluster.shape)  # torch.Size([100, 20])

        return out_f, (out_c_semantic, out_c_cluster), \
               (out_pool, out_layer3, out_layer2, out_layer1, out_conv1_3, out_conv1_2, out_conv1_1)


def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet20_hier(num_f_classes, num_c_classes, LDAM_net=False):
    return ResNet_s_hier(BasicBlock, [3, 3, 3], num_cls=num_f_classes, num_c_classes=num_c_classes, LDAM_net=LDAM_net)


def resnet32(num_f_classes=10, LDAM_net=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_f_classes=num_f_classes, LDAM_net=LDAM_net)


def resnet32_hier(num_cls, num_cls_c_semantic, LDAM_net=False):
    return ResNet_s_hier(BasicBlock, [5, 5, 5], num_cls, num_cls_c_semantic, LDAM_net)


def resnet32_hier_2c(num_f_classes, num_c_classes_semantic, num_c_classes_cluster, LDAM_net=False):
    return ResNet_s_hier_2c(BasicBlock, [5, 5, 5], num_f_classes, num_c_classes_semantic, num_c_classes_cluster, LDAM_net)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet44_hier(num_f_classes, num_c_classes, LDAM_net=False):
    return ResNet_s_hier(BasicBlock, [7, 7, 7], num_cls=num_f_classes, num_c_classes=num_c_classes, LDAM_net=LDAM_net)


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet56_hier(num_f_classes, num_c_classes, LDAM_net=False):
    return ResNet_s_hier(BasicBlock, [9, 9, 9], num_cls=num_f_classes, num_c_classes=num_c_classes, LDAM_net=LDAM_net)


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet110_hier(num_f_classes, num_c_classes, LDAM_net=False):
    return ResNet_s_hier(BasicBlock, [18, 18, 18], num_cls=num_f_classes, num_c_classes=num_c_classes, LDAM_net=LDAM_net)


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def resnet1202_hier(num_f_classes, num_c_classes, LDAM_net=False):
    return ResNet_s_hier(BasicBlock, [200, 200, 200], num_cls=num_f_classes, num_c_classes=num_c_classes, LDAM_net=LDAM_net)
