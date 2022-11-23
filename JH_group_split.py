# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import time
import argparse
import JH_utile
import math
import numpy as np

net_info = { 0: ['cifar10', 'cifar100'],                    # dataset
             # 1: ['resnet', 'wide_resnet', 'preact_resnet', 'pyramid_net'], # net_type
             # 2: ['depth', 'widen_factor', 'alpha', 'bottleneck'],    # net_component
             # 3: ['False', 'True']                           # bool variable list
           }


######################################################################
# # Parameter setting
# Path for save checkpoint
class_score_file = 'pyramid_net_SD-H1-D_272_alpha200_pl0.5-G_100_bottleneck_class_score'
base_save_path              = '/home/jinho/0_project_temp/'
class_score_path            = base_save_path + 'class_score/' + net_info[0][1] + os.sep
result_group_split_path     = base_save_path + 'result_group_split/' + net_info[0][1] + os.sep

for i, dataset in enumerate(net_info[0]):
    result_group_split_path_list = ['result_group_split',
                                    dataset]
    _ = JH_utile.make_Dir(base_save_path, result_group_split_path_list)



class_score = torch.load(class_score_path + class_score_file + '.pth')
args = class_score['args']



use_cuda        = torch.cuda.is_available()
# network device define
if use_cuda:
    device = torch.device('cuda:0')
    device_location = 'cuda:0'
    # data_parallel_device = [1]
    data_parallel_device = range(torch.cuda.device_count())
else:
    device = torch.device('cpu')
    device_location = 'cpu'



# number_group_target_list = [1, 3, 2]
number_group_target_list = [1, 5, 5]
h_level             = len(number_group_target_list)  # hierarchical level
lamda               = [1, 1, 5]

split_target = 'train'
num_epochs_split = 15000

######################################################################
# # load class score

train_set_class_score   = class_score['train_set_class_score']
train_set_label         = class_score['train_set_label']
test_set_class_score    = class_score['test_set_class_score']
test_set_label          = class_score['test_set_label']
num_classes             = class_score['num_classes'][0]
# num_classes = num_classes[0]


# entire info
# initialize group label info dictionary

ori_label = np.arange(num_classes)
ori_label = ori_label.tolist()
tree_history_init = []
for i in range(num_classes):
    tree_history_init.append(0)


group_label_info    = {'h_level'              : h_level,
                       'number_group_target_list'  : number_group_target_list,   # target number of split in each hierarchical level
                       'lamda'                : lamda,
                       # 'tree_history'         : [tree_history_init],
                       'group_label'          : [ori_label],   # h_level * num_classes
                       'num_group'            : [1],   # number of group in each hierarchical level
                       'num_classes_in_group' : [[num_classes]],   # number of class in each group at each hierarchical level
                       'class_list_in_group'  : [[ori_label]],   # h_level * num_group * class_list
                       'args'                 : args
                       }

lamda1 = lamda[0]
lamda2 = lamda[1]
lamda3 = math.pow(0.1, lamda[2])
lamda3 = round(lamda3, lamda[2])


num_train_set           = train_set_class_score.size()[0]
num_each_class_train    = num_train_set / num_classes
num_test_set            = test_set_class_score.size()[0]
num_each_class_test     = num_test_set / num_classes


# ============================================================== #
#                             test                               #
# test_temp = class_list_in_group[0][0]
#
# print(test_temp)



#                           test end                             #
# ============================================================== #


if split_target == 'train':
    num_data            = num_train_set
    data_class_score    = train_set_class_score
    data_label          = train_set_label
elif split_target == 'test':
    num_data            = num_test_set
    data_class_score    = test_set_class_score
    data_label          = test_set_label
else:
    assert split_target == 'train' or split_target == 'test', 'Error: Target is not defined!'


# 나중에 test_score_save 에서 라벨이 리스트로 저장되도록 수정 필요함.
data_label_list = []
for i in range(len(data_label)):
    temp_data_label = data_label[i]
    data_label_list.append(temp_data_label.data.item())

data_label = data_label_list


start_time = time.time()
for h in range(h_level-1):


    num_g_accu = 0
    temp_group_label          = torch.zeros(num_classes, dtype=torch.int64)
    # temp_num_group            = []
    temp_num_classes_in_group = []
    temp_class_list_in_group  = []

    num_classes_split_in_group = []
    class_list_split_in_group  = []

    for g in range(group_label_info['num_group'][h]):

        # data set setting
        if split_target == 'train':
            num_data = num_train_set
        elif split_target == 'test':
            num_data = num_test_set
        else:
            assert split_target == 'train' or split_target == 'test', 'Error: Target is not defined!'

        class_list = group_label_info['class_list_in_group'][h][g]
        num_class_list = len(class_list)
        # num_class_list = group_label_info['num_classes_in_group'][h][g]
        assert num_class_list == group_label_info['num_classes_in_group'][h][g], 'Error: num_class'


        temp_data_set_in_g = []
        temp_label_set_in_g = []

        mean_score_data = torch.zeros(num_class_list, num_classes)
        count_each_class = torch.zeros(num_class_list)

        for n_c in range(num_class_list):
            for n_d in range(num_data):
                if class_list[n_c] == data_label[n_d]:
                    temp_data_set_in_g.append(data_class_score[n_d])
                    temp_label_set_in_g.append(data_label[n_d])

                    mean_score_data[n_c]   += data_class_score[n_d]
                    count_each_class[n_c]   = count_each_class[n_c] + 1

        for n_c in range(num_class_list):
            if count_each_class[n_c] != 0:
                mean_score_data[n_c] = mean_score_data[n_c] / count_each_class[n_c]

        num_group_target = number_group_target_list[h + 1]
        split_module = JH_utile.GroupAssignmentLoss_V2(num_class_list, num_group_target)
        ## cuda 를 쓰면 cpu 보다 느림.
        # if use_cuda:
        #     split_module.to(device)
        #     net = nn.DataParallel(split_module, device_ids =data_parallel_device)
        #     cudnn.benchmark = True
        #     mean_score_data = mean_score_data.to(device)
        optimizer = optim.SGD(split_module.parameters(), lr=0.1, momentum=0.9)

        split_loss = 0


        for epoch in range(num_epochs_split):

            optimizer.zero_grad()
            disjoint_loss   = split_module(mean_score_data)
            overlap_loss    = split_module.overlap_loss()
            balance_loss    = split_module.balance_loss()
            total_loss      = lamda1 * disjoint_loss + lamda2 * overlap_loss + lamda3 * balance_loss

            total_loss.backward()
            optimizer.step()

            print('\n| epoch: [%5d/%5d], h_level: %2d group_number: %3d lamda1: %2d, lamda2: %2d, lamda3: %2d\n'
                  '|d_loss: %.6f, o_loss: %.6f, b_loss: %.6f'
                  % (epoch, num_epochs_split, h, g, lamda1, lamda2, lamda[2],
                     disjoint_loss, overlap_loss, balance_loss))

        # save group list
        p = split_module.p
        p_group_prob, p_group_index = torch.max(p, 1)


        for g_idx in range(num_group_target):
            class_list_split = []
            for i in range(num_class_list):
                if p_group_index[i] == g_idx:
                    class_list_split.append(class_list[i])
            if len(class_list_split) != 0:
                temp_num_classes_in_group.append(len(class_list_split))
                temp_class_list_in_group.append(class_list_split)


        # calculate loss value after split
        p_replace = torch.zeros(num_class_list, num_group_target)
        for i in range(num_class_list):
            p_replace[i, p_group_index[i]] = 1

        with torch.no_grad():
            # split_module.p = p_replace
            disjoint_loss_after_split   = split_module(mean_score_data, p_replace)
            overlap_loss_after_split    = split_module.overlap_loss(p_replace)
            balance_loss_after_split    = split_module.balance_loss(p_replace)

        print('check')


        num_g_accu += len(num_classes_split_in_group)


    temp_num_group = len(temp_num_classes_in_group)
    group_label_info['num_group'           ].append(temp_num_group)
    group_label_info['num_classes_in_group'].append(temp_num_classes_in_group)
    group_label_info['class_list_in_group' ].append(temp_class_list_in_group)

    for n_g in range(temp_num_group):
        for n_c_g in range(len(temp_class_list_in_group[n_g])):
            group_label_idx = temp_class_list_in_group[n_g][n_c_g]
            temp_group_label[group_label_idx] = n_g


    group_label_info['group_label'].append(temp_group_label.tolist())


assert args.dataset == 'cifar100' or args.dataset =='cifar10', 'Check: dataset'
if args.dataset == 'cifar100':
    num_classes = [100]
elif args.dataset == 'cifar10':
    num_classes = [10]
file_name = JH_utile.get_filename_for_split(args, group_label_info, num_classes)

torch.save(group_label_info, result_group_split_path + file_name + '.pth')


epoch_time = time.time() - start_time

print(epoch_time)
print('check')
