# -*-coding:utf-8-*-
# from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from networks.ShakeDrop import ShakeDrop





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
    outchannel_ratio = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, p_shakedrop=1.0):
        super(BasicBlock, self).__init__()
        self.bn1        = nn.BatchNorm2d(in_planes)
        self.conv1      = conv3x3(in_planes, planes, stride)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv2      = conv3x3(planes, planes)
        self.bn3        = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]

        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = Variable(torch.cuda.FloatTensor(batch_size,
                               residual_channel - shortcut_channel,
                               featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out

class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, p_shakedrop=1.0):
        super(Bottleneck, self).__init__()
        self.bn1        = nn.BatchNorm2d(in_planes)
        self.conv1      = conv1x1(in_planes, planes)
        self.bn2        = nn.BatchNorm2d(planes)
        # self.conv2      = conv3x3(planes, (planes*1), stride=stride)
        self.conv2      = conv3x3(planes, planes, stride=stride)
        # self.bn3        = nn.BatchNorm2d((planes*1))
        self.bn3        = nn.BatchNorm2d(planes)
        # self.conv3      = conv1x1((planes*1), planes * Bottleneck.outchannel_ratio)
        self.conv3      = conv1x1(planes, planes * Bottleneck.outchannel_ratio)
        self.bn4        = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.downsample = downsample
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn3(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = Variable(torch.cuda.FloatTensor(batch_size,
                               residual_channel - shortcut_channel, featuremap_size[0],
                               featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class PyramidNet_ShakeDrop(nn.Module):
    def __init__(self, depth_list, num_classes_list, dataset, alpha, bottleneck=False, pl=0.5):
        super(PyramidNet_ShakeDrop, self).__init__()

        depth = depth_list[0]
        num_classes = num_classes_list[0]

        self.dataset = dataset
        self.pl = pl
        if self.dataset.startswith('cifar'):
            self.in_planes = 16

            if bottleneck == False:
                assert (depth - 2) % 6 == 0, \
                    'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
                n = (depth - 2) // 6
                block_type = BasicBlock

            else:
                assert (depth - 2) % 9 == 0, \
                    'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
                n = (depth - 2) // 9
                block_type = Bottleneck

            self.addrate = alpha / (3 * n * 1.0)

            # self.ps_shakedrop = [1. - (1.0 - (0.5 / (3 * n)) * (i + 1)) for i in range(3 * n)] # 아래 for 문과 같음
            self.ps_shakedrop = []
            self.ps_sd_index = 0
            for i in range(3 * n):
                self.ps_shakedrop.append(1. - (1.0 - (self.pl / (3 * n)) * (i + 1)))

            self.conv1  = conv3x3(3, self.in_planes)
            self.bn1    = nn.BatchNorm2d(self.in_planes)

            self.featuremap_dim = self.in_planes

            self.layer1 = self._make_layer(block_type, n)
            self.layer2 = self._make_layer(block_type, n, stride=2)
            self.layer3 = self._make_layer(block_type, n, stride=2)

            self.bn2 = nn.BatchNorm2d(self.in_planes)
            self.fc = nn.Linear(self.in_planes, num_classes)


        elif dataset == 'imagenet':
            block_type = {18: BasicBlock, 34: BasicBlock,
                          50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                      101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid depth for Pre-ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

            self.in_planes = 64
            self.block_number = sum(layers[depth])
            self.addrate = alpha / (self.block_number * 1.0)

            self.ps_shakedrop = []
            self.ps_sd_index = 0
            for i in range(self.block_number):
                self.ps_shakedrop.append(1. - (1.0 - (self.pl / self.block_number) * (i + 1)))

            self.in_planes = self.in_planes
            self.conv1  = conv7x7(3, self.in_planes, stride=2)
            self.bn1    = nn.BatchNorm2d(self.in_planes)

            self.featuremap_dim = self.in_planes
            self.layer1 = self._make_layer(block_type[depth], layers[depth][0])
            self.layer2 = self._make_layer(block_type[depth], layers[depth][1], stride=2)
            self.layer3 = self._make_layer(block_type[depth], layers[depth][2], stride=2)
            self.layer4 = self._make_layer(block_type[depth], layers[depth][3], stride=2)

            self.bn2 = nn.BatchNorm2d(self.in_planes)
            self.fc = nn.Linear(self.in_planes, num_classes)
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block_type, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block_type(self.in_planes, int(round(self.featuremap_dim)), stride, downsample,
                                 p_shakedrop=self.ps_shakedrop[self.ps_sd_index]))
        self.ps_sd_index = self.ps_sd_index + 1
        for i in range(1, blocks):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block_type(int(round(self.featuremap_dim)) * block_type.outchannel_ratio,
                                     int(round(temp_featuremap_dim)), 1, p_shakedrop=self.ps_shakedrop[self.ps_sd_index]))
            self.ps_sd_index = self.ps_sd_index + 1
            self.featuremap_dim = temp_featuremap_dim
        self.in_planes = int(round(self.featuremap_dim)) * block_type.outchannel_ratio

        return nn.Sequential(*layers)



    def forward(self, x):
        if self.dataset.startswith('cifar'):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.bn2(out)
            out = F.relu(out, inplace=True)
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.fc(out)

        elif self.dataset == 'imagenet':
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.bn2(out)
            out = F.relu(out, inplace=True)
            out = F.avg_pool2d(out, 7)
            out = out.view(out.size(0), -1)
            out = self.fc(out)

        return out




