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

    # def forward(self, x):
    #     out = self.conv1(F.relu(self.bn1(x)))
    #     out = self.conv2(F.relu(self.bn2(out)))
    #     out += self.shortcut(x)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth_list, num_classes_list, widen_factor, dropout_rate):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        depth = depth_list[0]
        num_classes = num_classes_list[0]

        assert ((depth-4)%6 ==0), 'wide_resnet depth should be 6n+4'
        n = (depth-4)//6
        w_f = widen_factor

        self.palnes = [16*w_f, 32*w_f, 64*w_f]

        self.conv1 = conv3x3(3, self.in_planes)
        self.layer1 = self._make_layer(BasicBlock, self.in_planes, self.palnes[0], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(BasicBlock, self.in_planes, self.palnes[1], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.in_planes, self.palnes[2], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(self.palnes[2], momentum=0.9)
        self.linear = nn.Linear(self.palnes[2], num_classes)
        # self.conv1 = conv3x3(3,self.nStages[0])
        # self.layer1 = self._make_layer(BasicBlock, self.nStages[1], n, dropout_rate, stride=1)
        # self.layer2 = self._make_layer(BasicBlock, self.nStages[2], n, dropout_rate, stride=2)
        # self.layer3 = self._make_layer(BasicBlock, self.nStages[3], n, dropout_rate, stride=2)
        # self.bn1 = nn.BatchNorm2d(self.nStages[3], momentum=0.9)
        # self.linear = nn.Linear(self.nStages[3], num_classes)

    def _make_layer(self, block_type, in_planes, planes, blocks, dropout_rate, stride=1):
        layers = []
        layers.append(block_type(in_planes, planes, dropout_rate, stride=stride))
        self.in_planes = planes

        for i in range(1, blocks):
            layers.append(block_type(self.in_planes, planes, dropout_rate))
        return nn.Sequential(*layers)


    # def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, dropout_rate, stride))
    #         self.in_planes = planes
    #
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out



class Wide_ResNet_H2(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_H2, self).__init__()
        self.in_planes = 16

        assert len(num_classes) == 2, 'Check: is not same - h level  len(num_classes) - '
        assert ((depth-4)%6 ==0), 'wide_resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._make_layer(BasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(BasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(BasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear_h1 = nn.Linear(nStages[3], num_classes[0])
        self.linear_h2 = nn.Linear(nStages[3], num_classes[1])


    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out_h1 = self.linear_h1(out)
        out_h2 = self.linear_h2(out)

        return out_h1, out_h2




class Wide_ResNet_H3(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_H3, self).__init__()
        self.in_planes = 16

        assert len(num_classes) == 3, 'Check: is not same - h level  len(num_classes) - '
        assert ((depth-4)%6 ==0), 'wide_resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._make_layer(BasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(BasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(BasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear_h1 = nn.Linear(nStages[3], num_classes[0])
        self.linear_h2 = nn.Linear(nStages[3], num_classes[1])
        self.linear_h3 = nn.Linear(nStages[3], num_classes[2])

    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out_h1 = self.linear_h1(out)
        out_h2 = self.linear_h2(out)
        out_h3 = self.linear_h3(out)

        return out_h1, out_h2, out_h3


class Wide_ResNet_H4(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_H4, self).__init__()
        self.in_planes = 16

        assert len(num_classes) == 4, 'Check: is not same - h level  len(num_classes) - '
        assert ((depth-4)%6 ==0), 'wide_resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._make_layer(BasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(BasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(BasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear_h1 = nn.Linear(nStages[3], num_classes[0])
        self.linear_h2 = nn.Linear(nStages[3], num_classes[1])
        self.linear_h3 = nn.Linear(nStages[3], num_classes[2])
        self.linear_h4 = nn.Linear(nStages[3], num_classes[3])


    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out_h1 = self.linear_h1(out)
        out_h2 = self.linear_h2(out)
        out_h3 = self.linear_h3(out)
        out_h4 = self.linear_h4(out)

        return out_h1, out_h2, out_h3, out_h4


#
# def get_filename(args):
#     if (args.disjoint == True):
#         file_name = args.net_type + '-' + str(args.depth) + 'x' + str(args.widen_factor) + '_disjoint'
#     else:
#         file_name = args.net_type + '-' + str(args.depth) + 'x' + str(args.widen_factor)
#
#     return file_name
#
# def get_filename_for_split(args, group_label_info):
#
#     h_level             = group_label_info['h_level']
#     num_group           = group_label_info['num_group']
#     number_split_target = group_label_info['number_split_target']
#     lamda               = group_label_info['lamda']
#
#     str_num_group = '_G'
#     for n_g_l in range(len(num_group)):
#         str_num_group = str_num_group + '-{G:02d}'.format(G=num_group[n_g_l])
#
#     str_split_target = '_S'
#     for s_t in range(len(number_split_target)):
#         str_split_target = str_split_target + '-{S:02d}'.format(S=number_split_target[s_t])
#
#     str_lamda_value = '_L'
#     lamda3_str = '-(10e-{l3})'.format(l3=lamda[2])
#     for l_v in range(len(lamda) - 1):
#         str_lamda_value = str_lamda_value + '-{L:02d}'.format(L=lamda[l_v])
#     str_lamda_value = str_lamda_value + lamda3_str
#
#     if (args.disjoint == True):
#         file_name = args.net_type + '-' + str(args.depth) + 'x' + str(args.widen_factor) \
#                     + '_h{H}'.format(H=h_level) + str_num_group + str_split_target + str_lamda_value + '_disjoint'
#     else:
#         file_name = args.net_type + '-' + str(args.depth) + 'x' + str(args.widen_factor) \
#                     + '_h{H}'.format(H=h_level) + str_num_group + str_split_target + str_lamda_value
#
#     return file_name
#
#
# def load_param(args, checkpoint):
#     # Load checkpoint data
#     print("| Resuming from checkpoint...")
#     # checkpoint = torch.load(saved_filename)
#
#     args.lr             = checkpoint['lr']
#     args.depth          = checkpoint['depth']
#     args.widen_factor   = checkpoint['widen_factor']
#     args.dropout        = checkpoint['dropout']
#     args.lr_drop_epoch  = checkpoint['lr_drop_epoch']
#     args.start_epoch    = checkpoint['best_epoch']
#     best_state_dict = checkpoint['best_state_dict']
#     best_acc = checkpoint['best_acc']
#
#     return args, best_state_dict, best_acc
#
#
# def get_network_for_split(args, num_classes):
#     if (args.net_type == 'wide_resnet'):
#         if len(num_classes) == 1:
#             net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
#         elif len(num_classes) == 2:
#             net = Wide_ResNet_H2(args.depth, args.widen_factor, args.dropout, num_classes)
#         elif len(num_classes) == 3:
#             net = Wide_ResNet_H3(args.depth, args.widen_factor, args.dropout, num_classes)
#         elif len(num_classes) == 4:
#             net = Wide_ResNet_H4(args.depth, args.widen_factor, args.dropout, num_classes)
#
#         else:
#             assert len(num_classes) == 2 or len(num_classes) == 3 or len(num_classes) == 4, 'Check: please check the h_level'
#
#     else:
#         print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
#         sys.exit(0)
#
#     return net
#
#
# def get_network(args, num_classes):
#     if (args.net_type == 'wide_resnet'):
#         net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
#     else:
#         print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
#         sys.exit(0)
#
#     return net
#
#
# def learning_rate(init, epoch, lr_drop_epoch):
#     optim_factor = 0
#     if(epoch > lr_drop_epoch[2]):
#         optim_factor = 3
#
#     elif(epoch > lr_drop_epoch[1]):
#         optim_factor = 2
#
#     elif(epoch > lr_drop_epoch[0]):
#         optim_factor = 1
#
#
#     return init*math.pow(0.2, optim_factor)


if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())


