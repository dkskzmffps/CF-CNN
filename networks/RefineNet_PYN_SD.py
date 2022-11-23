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


class RefineNet_PyramidNet_ShakeDrop(nn.Module):
    def __init__(self, depth_list, num_classes_list, dataset, alpha, bottleneck=False, pl=0.5):
        super(RefineNet_PyramidNet_ShakeDrop, self).__init__()

        depth = depth_list[0]
        num_classes = num_classes_list[0]

        self.dataset = dataset
        self.pl = pl
        self.pl2 = 0.05
        if self.dataset.startswith('cifar'):
            self.in_planes = 16

            if bottleneck == False:
                assert (depth_list[0] - 2) % 6 == 0, \
                    'When use basicblock, main net. depth should be 6n+2, e.g.20, 32, 44, 56, 110, 1202'
                assert (depth_list[1]) % 2 == 0 and (depth_list[1]) // 2 >= 2, \
                    'When use basicblock, sub net. depth should be 2n, e.g. 18, 26, 38, 44, 54, 108, 1200 \n' \
                    'depth should be greather than 4 (2n)'
                n_main = (depth_list[0] - 2) // 6
                n_sub1 = depth_list[1] // 2  # won block has 2 layer
                block_type = BasicBlock

            else:
                assert (depth_list[0] - 2) % 9 == 0, \
                    'When use bottleneck, main net. depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
                assert (depth_list[1]) % 3 == 0 and depth_list[1] // 3 >= 2, \
                    'When use bottleneck, sub net. depth should be 9n, e.g. 18, 27, 39, 45, 54, 108, 1197 \n' \
                    'depth should be greather than 6 (3n)'

                n_main = (depth_list[0] - 2) // 9
                n_sub1 = depth_list[1] // 3  # won block has 3 layer
                block_type = Bottleneck

            self.addrate        = alpha / (3 * n_main * 1.0)
            self.addrate_sub1   = 0

            merging_blocks           = 2
            out_planes_block3        = self.in_planes + (self.addrate * n_main * 3)  # number of featuremap_dim of last layer
            out_featuremap_dim       = int(round(out_planes_block3)) * block_type.outchannel_ratio
            self.in_planes_sub1      = self.in_planes
            self.featuremap_dim      = self.in_planes
            self.featuremap_dim_sub1 = self.in_planes


            self.ps_shakedrop       = []
            self.ps_shakedrop_sub1  = []
            self.ps_sd_index        = 0
            self.ps_sd_index_sub1   = 0

            # self.ps_shakedrop = [1. - (1.0 - (0.5 / (3 * n)) * (i + 1)) for i in range(3 * n)] # 아래 for 문과 같음
            for i in range((3 * n_main) + merging_blocks):
                self.ps_shakedrop.append(1. - (1.0 - (self.pl / ((3 * n_main) + merging_blocks)) * (i + 1)))
            # for i in range((3 * n_main) + merging_blocks):
            #     self.ps_shakedrop.append(1. - (1.0 - (self.pl / (3 * n_main) + merging_blocks) * (i + 1)))

            self.conv0  = conv3x3(3, self.in_planes)
            self.bn0    = nn.BatchNorm2d(self.in_planes)

            self.layer1 = self._make_layer(block_type, n_main)

            self.featuremap_dim_sub1 = self.featuremap_dim
            self.addrate_sub1       = (out_planes_block3 - self.featuremap_dim_sub1) / (1 * n_sub1 * 1.0)
            self.in_planes_sub1     = self.in_planes
            self.ps_sd_index_sub1   = self.ps_sd_index

            self.layer1_sub1 = self._make_layer_sub(block_type, math.floor(n_sub1//2), stride=2)
            self.layer1_sub2 = self._make_layer_sub(block_type, n_sub1 - math.floor(n_sub1 // 2), stride=2)
            self.bn1         = nn.BatchNorm2d(self.in_planes_sub1)
            self.fc1         = nn.Linear(self.in_planes_sub1, num_classes_list[0])
            assert self.in_planes_sub1 == out_featuremap_dim, 'please check Addrate and layer number'

            self.layer2 = self._make_layer(block_type, n_main, stride=2)

            self.featuremap_dim_sub1 = self.featuremap_dim
            self.addrate_sub1        = (out_planes_block3 - self.featuremap_dim_sub1) / (1 * n_sub1 * 1.0)
            self.in_planes_sub1      = self.in_planes
            self.ps_sd_index_sub1    = self.ps_sd_index

            self.layer2_sub1 = self._make_layer_sub(block_type, n_sub1, stride=2)
            self.bn2 = nn.BatchNorm2d(self.in_planes_sub1)
            self.fc2 = nn.Linear(self.in_planes_sub1, num_classes_list[1])
            assert self.in_planes_sub1 == out_featuremap_dim, 'please check Addrate and layer number'

            self.layer3 = self._make_layer(block_type, n_main, stride=2)
            self.bn3 = nn.BatchNorm2d(self.in_planes)
            self.fc3 = nn.Linear(self.in_planes, num_classes_list[2])
            assert self.in_planes == out_featuremap_dim, 'please check Addrate and layer number'

            self.ps_sd_index = self.ps_sd_index-1

            self.layer_merging = self._make_layer_merging(block_type, merging_blocks)
            assert self.in_planes == out_featuremap_dim, 'please check Addrate and layer number'
            self.bn_merging = nn.BatchNorm2d(self.in_planes)
            self.fc_merging = nn.Linear(self.in_planes, num_classes_list[2])

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
                                     int(round(temp_featuremap_dim)), 1,
                                     p_shakedrop=self.ps_shakedrop[self.ps_sd_index]))
            self.ps_sd_index = self.ps_sd_index + 1
            self.featuremap_dim = temp_featuremap_dim
        self.in_planes = int(round(self.featuremap_dim)) * block_type.outchannel_ratio

        return nn.Sequential(*layers)

    def _make_layer_sub(self, block_type, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim_sub1 = self.featuremap_dim_sub1 + self.addrate_sub1
        layers.append(block_type(self.in_planes_sub1, int(round(self.featuremap_dim_sub1)), stride, downsample,
                                 p_shakedrop=self.ps_shakedrop[self.ps_sd_index_sub1]))
        self.ps_sd_index_sub1 = self.ps_sd_index_sub1 + 1
        for i in range(1, blocks):
            temp_featuremap_dim = self.featuremap_dim_sub1 + self.addrate_sub1
            layers.append(block_type(int(round(self.featuremap_dim_sub1)) * block_type.outchannel_ratio,
                                     int(round(temp_featuremap_dim)), 1,
                                     p_shakedrop=self.ps_shakedrop[self.ps_sd_index_sub1]))
            self.ps_sd_index_sub1 = self.ps_sd_index_sub1 + 1
            self.featuremap_dim_sub1 = temp_featuremap_dim
        self.in_planes_sub1 = int(round(self.featuremap_dim_sub1)) * block_type.outchannel_ratio

        return nn.Sequential(*layers)


    def _make_layer_merging(self, block_type, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        layers.append(block_type(self.in_planes, int(round(self.featuremap_dim)), stride, downsample,
                                 p_shakedrop=self.ps_shakedrop[self.ps_sd_index]))
        self.ps_sd_index = self.ps_sd_index + 1
        for i in range(1, blocks):
            temp_featuremap_dim = self.featuremap_dim
            layers.append(block_type(int(round(self.featuremap_dim)) * block_type.outchannel_ratio,
                                     int(round(temp_featuremap_dim)), 1,
                                     p_shakedrop=self.ps_shakedrop[self.ps_sd_index]))
            self.ps_sd_index = self.ps_sd_index + 1
            self.featuremap_dim = temp_featuremap_dim
        self.in_planes = int(round(self.featuremap_dim)) * block_type.outchannel_ratio

        return nn.Sequential(*layers)


    def forward(self, x):
        if self.dataset.startswith('cifar'):
            out = self.conv0(x)
            out = self.bn0(out)
            out = self.layer1(out)

            sub1 = self.layer1_sub1(out)
            sub1 = self.layer1_sub2(sub1)
            sub1 = self.bn1(sub1)
            sub1 = F.relu(sub1, inplace=True)
            out_sub1 = F.avg_pool2d(sub1, 8)
            out_sub1 = out_sub1.view(out_sub1.size(0), -1)
            out_sub1 = self.fc1(out_sub1)

            out = self.layer2(out)
            sub2 = self.layer2_sub1(out)
            sub2 = self.bn2(sub2)
            sub2 = F.relu(sub2, inplace=True)
            out_sub2 = F.avg_pool2d(sub2, 8)
            out_sub2 = out_sub2.view(out_sub2.size(0), -1)
            out_sub2 = self.fc2(out_sub2)

            sub3 = self.layer3(out)
            sub3 = self.bn3(sub3)
            sub3 = F.relu(sub3, inplace=True)
            out_sub3 = F.avg_pool2d(sub3, 8)
            out_sub3 = out_sub3.view(out_sub3.size(0), -1)
            out_sub3 = self.fc3(out_sub3)

            out = sub1 + sub2 + sub3
            out = self.layer_merging(out)
            out = self.bn_merging(out)
            out = F.relu(out, inplace=True)
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.fc_merging(out)

        return out_sub1, out_sub2, out_sub3, out




