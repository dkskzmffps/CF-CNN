# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import math

import sys
import numpy as np
import os

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

# def weight_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         init.constant_(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         w = torch.rand(m.weight.data.size())
#         m.weight.data = w
#         init.constant_(m.bias, 0)


def weight_init(m):
    classname = m.__class__.__name__
    # != : 같지 않다면. ex) 1!=2  --> True,  1!=1  --> False
    # find() 함수와 같이 쓰이는 경우 원본 문자열 안에 매개변수로 입력한 문자열이 존재하지 않으면 -1을 반환
    # 존재하는 문자열이 있을경우 문자열 위치를 반환
    # ex) a = 'hello', 1) a.find('ll') --> 2, 2) a.find('H') --> 0, 3) a.find('J') --> -1
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth_list, num_classes_list, dataset, widen_factor, dropout_rate):
        super(Wide_ResNet, self).__init__()
        self.dataset = dataset
        self.in_planes = 16
        self.in_planes_sub = 16

        if self.dataset.startswith('cifar'):
            assert (depth_list[0] - 4) % 6 == 0, 'When use basicblock, main net. depth should be 6n+2'
            assert (depth_list[1] % 2) == 0 and (depth_list[1]) // 2 >= 2, \
                'When use basicblock, sub net. depth should be 2n, \n ' \
                'depth should be greather than 4 (2n)'
            block_type = BasicBlock
            n_main = (depth_list[0] - 4 ) // 6
            n_sub1 = depth_list[1] // 2
            w_f = widen_factor
            self.palnes = [16*w_f, 32*w_f, 64*w_f]

            self.conv1          = conv3x3(3, self.in_planes)
            self.layer1         = self._make_layer(BasicBlock, self.in_planes, self.palnes[0], n_main, dropout_rate)

            self.in_planes_sub  = self.in_planes
            self.layer1_sub1    = self._make_layer_sub(block_type, self.in_planes_sub, self.palnes[1], math.floor(n_sub1//2),dropout_rate, stride=2)
            self.layer1_sub2    = self._make_layer_sub(block_type, self.in_planes_sub, self.palnes[2], n_sub1-math.floor(n_sub1//2),dropout_rate, stride=2)
            self.bn1            = nn.BatchNorm2d(self.palnes[2])
            self.fc1            = nn.Linear(self.palnes[2], num_classes_list[0])

            self.layer2         = self._make_layer(BasicBlock, self.in_planes, self.palnes[1], n_main, dropout_rate, stride=2)
            self.in_planes_sub  = self.in_planes
            self.layer2_sub1    = self._make_layer_sub(block_type, self.in_planes_sub, self.palnes[2], n_sub1, dropout_rate, stride=2)
            self.bn2            = nn.BatchNorm2d(self.palnes[2])
            self.fc2            = nn.Linear(self.palnes[2], num_classes_list[1])

            self.layer3         = self._make_layer(BasicBlock, self.in_planes, self.palnes[2], n_main, dropout_rate, stride=2)
            self.bn3            = nn.BatchNorm2d(self.palnes[2])
            self.fc3            = nn.Linear(self.palnes[2], num_classes_list[2])


            self.layer_merging  = self._make_layer(block_type, self.in_planes, self.palnes[2], 1, dropout_rate)
            self.bn_merging     = nn.BatchNorm2d(self.palnes[2])
            self.fc_merging     = nn.Linear(self.palnes[2], num_classes_list[2])


    def _make_layer(self, block_type, in_planes, planes, blocks, dropout_rate, stride=1):
        layers = []
        layers.append(block_type(in_planes, planes, dropout_rate, stride=stride))
        self.in_planes = planes

        for i in range(1, blocks):
            layers.append(block_type(self.in_planes, planes, dropout_rate))
        return nn.Sequential(*layers)

    def _make_layer_sub(self, block_type, in_planes, planes, blocks, dropout_rate, stride=1):
        layers = []
        layers.append(block_type(in_planes, planes, dropout_rate, stride=stride))
        self.in_planes_sub = planes

        for i in range(1, blocks):
            layers.append(block_type(self.in_planes_sub, planes, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
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



if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())


