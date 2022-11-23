# -*-coding:utf-8-*-

import torch.nn as nn
import math
import torch.nn.functional as F


def conv1x1 (in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3 (in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv7x7 (in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2./n))
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1      = conv3x3(in_planes, planes, stride)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = nn.BatchNorm2d(planes)
        # self.relu       = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out, inplace=True)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1  = conv1x1(in_planes, planes)
        self.bn1    = nn.BatchNorm2d(planes)
        self.conv2  = conv3x3(planes, planes, stride=stride)
        self.bn2    = nn.BatchNorm2d(planes)
        self.conv3  = conv1x1(planes, planes * Bottleneck.expansion)
        self.bn3    = nn.BatchNorm2d(planes*Bottleneck.expansion)
        # self.relu   = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class RefineNet_ResNet(nn.Module):
    def __init__(self, depth_list, num_classes_list, dataset, bottleneck=False):
        super(RefineNet_ResNet, self).__init__()

        print('| Apply bottleneck: {TF}'.format(TF=bottleneck))

        self.dataset = dataset
        self.in_planes = 16
        self.in_planes_sub = 16
        # except_layer = []
        # h_classes = len(num_classes_list) # h_level 이 낮을 수록 class수가 적다

        if self.dataset.startswith('cifar'):
            if bottleneck == False:
                assert (depth_list[0] - 2) % 6 == 0, \
                    'When use basicblock, main net. depth should be 6n+2, e.g.20, 32, 44, 56, 110, 1202'
                assert (depth_list[1]) % 2 == 0 and (depth_list[1]) // 2 >= 2, \
                    'When use basicblock, sub net. depth should be 2n, e.g. 18, 26, 38, 44, 54, 108, 1200 \n' \
                    'depth should be greather than 4 (2n)'
                block_type = BasicBlock
                n_main = (depth_list[0] - 2) // 6
                n_sub1 = depth_list[1] // 2

            else:
                assert (depth_list[0] - 2) % 9 == 0, \
                    'When use bottleneck, main net. depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
                assert (depth_list[1]) % 3 == 0 and depth_list[1] // 3 >= 2, \
                    'When use bottleneck, sub net. depth should be 9n, e.g. 18, 27, 39, 45, 54, 108, 1197 \n' \
                    'depth should be greather than 6 (3n)'
                block_type = Bottleneck
                n_main = (depth_list[0] - 2) // 9
                n_sub1 = depth_list[1] // 3


            self.conv1 = conv3x3(3, self.in_planes)
            self.bn1     = nn.BatchNorm2d(self.in_planes)
            # self.relu    = nn.ReLU(inplace=True)
            self.layer1  = self._make_layer(block_type, self.in_planes, 16, n_main)
            self.in_planes_sub = self.in_planes

            self.layer1_sub1 = self._make_layer_sub(block_type, self.in_planes_sub, 32, math.floor(n_sub1 // 2), stride=2)
            self.layer1_sub2 = self._make_layer_sub(block_type, self.in_planes_sub, 64, n_sub1 - math.floor(n_sub1 // 2), stride=2)
            self.fc1 = nn.Linear(64*block_type.expansion, num_classes_list[0])

            self.layer2  = self._make_layer(block_type, self.in_planes, 32, n_main, stride=2)
            self.in_planes_sub = self.in_planes

            self.layer2_sub1 = self._make_layer_sub(block_type, self.in_planes_sub, 64, n_sub1, stride=2)
            self.fc2 = nn.Linear(64 * block_type.expansion, num_classes_list[1])


            self.layer3 = self._make_layer(block_type, self.in_planes, 64, n_main, stride=2)
            self.fc3 = nn.Linear(64 * block_type.expansion, num_classes_list[2])

            self.layer_merging = self._make_layer(block_type, self.in_planes, 64, 2)

            # self.avgpool = nn.AvgPool2d(8)
            self.fc_merging =nn.Linear(64 * block_type.expansion, num_classes_list[2])

        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


    def _make_layer(self, block_type, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes * block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(in_planes, planes * block_type.expansion, stride=stride),
                nn.BatchNorm2d(planes * block_type.expansion),
            )

        layers = []
        layers.append(block_type(in_planes, planes, stride, downsample))
        self.in_planes = planes * block_type.expansion
        for i in range(1, blocks):
            layers.append(block_type(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _make_layer_sub(self, block_type, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes * block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(in_planes, planes * block_type.expansion, stride=stride),
                nn.BatchNorm2d(planes * block_type.expansion),
            )

        layers = []
        layers.append(block_type(in_planes, planes, stride, downsample))
        self.in_planes_sub = planes * block_type.expansion
        for i in range(1, blocks):
            layers.append(block_type(self.in_planes_sub, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        if self.dataset.startswith('cifar'):
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out, inplace=True)

            out = self.layer1(out)
            sub1 = self.layer1_sub1(out)
            sub1 = self.layer1_sub2(sub1)
            out_sub1 = F.avg_pool2d(sub1, 8)
            out_sub1 = out_sub1.view(out_sub1.size(0), -1)
            out_sub1 = self.fc1(out_sub1)


            out = self.layer2(out)
            sub2 = self.layer2_sub1(out)
            out_sub2 = F.avg_pool2d(sub2, 8)
            out_sub2 = out_sub2.view(out_sub2.size(0), -1)
            out_sub2 = self.fc2(out_sub2)


            sub3 = self.layer3(out)
            out_sub3 = F.avg_pool2d(sub3, 8)
            out_sub3 = out_sub3.view(out_sub3.size(0), -1)
            out_sub3 = self.fc3(out_sub3)


            out = sub1 + sub2 + sub3
            out = self.layer_merging(out)

            out = F.avg_pool2d(out, 8)
            # out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc_merging(out)

        elif self.dataset == 'imagenet':
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out, inplace=True)
            # out = self.maxpool(out)
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 7)
            out = out.view(out.size(0), -1)
            out = self.fc(out)

        return out_sub1, out_sub2, out_sub3, out