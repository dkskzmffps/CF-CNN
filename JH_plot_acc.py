from __future__ import print_function

import argparse
import os
import torch

# from wide_resnet import *
# from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
import torch
import copy
import JH_utile


checkpoint_path = '/home/jinho/0_project_temp/checkpoint_for_plot2/'

net_info = { 0: ['cifar100'],                    # dataset
             # 1: ['resnet', 'wide_resnet', 'preact_resnet', 'pyramid_net','RefineNet_PRN'], # net_type
             # 1: ['resnet', 'preact_resnet', 'RefineNet_RN', 'RefineNet_PRN'], # net_type
             1: ['wide_resnet', 'preact_resnet', 'pyramid_net_SD', 'RefineNet_WRN', 'RefineNet_PRN', 'RefineNet_PYN_SD'], # net_type
             2: ['depth', 'widen_factor', 'alpha', 'bottleneck'],    # net_component
             3: ['False', 'True']                           # bool variable list
           }

text_info = { 0: ['CIFAR100'],          # dataset_name
              # 1: ['ResNet', 'Pre-ResNet', 'RefineNet_RN', 'RefineNet_PRN'],   # net_name, note: order of net name have to same with net_type in the net_info
              1: ['WRN', 'Pre-ResNet', 'PYN_SD', 'ML_WRN', 'ML_Pre-ResNet', 'ML_PYN_SD'],
              2: ['depth', 'width x ', 'alpha = ', 'bottleneck'],
              3: ['Top1 Error', 'Top5 Error']
            }

checkpoint_filename_list = []
for i, dataset in enumerate(net_info[0]): # net_info[0] : dataset list
    checkpoint_filename_list.append([])
    file_path = checkpoint_path + dataset + '/'
    checkpoint_list_temp = os.listdir(file_path)
    checkpoint_list_temp.sort()

    for j in range(len(checkpoint_list_temp)):
        if 'best' in checkpoint_list_temp[j]:
            continue
        else:
            checkpoint_filename_list[i].append(checkpoint_list_temp[j])


legend_list     = []
checkpoint_list = []

#=====================================================================================================================#
# Device setting =====================================================================================================#

use_cuda        = torch.cuda.is_available()
# use_cuda = False
# network device define
if use_cuda:
    device = torch.device('cuda:0')
    device_location = 'cuda:0'
    # data_parallel_device = [1]
    data_parallel_device = range(torch.cuda.device_count())
else:
    device = torch.device('cpu')
    device_location = 'cpu'

for i, dataset in enumerate(net_info[0]): # net_info[0] : dataset list
    legend_list.append([])
    checkpoint_list.append([])
    file_path = checkpoint_path + dataset + '/'
    for j in range(len(checkpoint_filename_list[i])):
        checkpoint = torch.load(file_path+checkpoint_filename_list[i][j],map_location=device_location)
        del checkpoint['state_dict']
        checkpoint_list[i].append(checkpoint)

    # # ======================================================================================= # #
    # # ==================================checkpoint sorting=================================== # #
    sort_loop_end = len(checkpoint_list[i]) - 1
    for j in range(len(checkpoint_list[i]) - 1):

        for k in range(sort_loop_end):
            plot_acc_info1 = checkpoint_list[i][k]['plot_acc_info']
            plot_acc_info2 = checkpoint_list[i][k+1]['plot_acc_info']
            net_type_index1 = net_info[1].index(plot_acc_info1['net_type'])
            net_type_index2 = net_info[1].index(plot_acc_info2['net_type'])

            if net_type_index1 != net_type_index2:
                if net_type_index1 > net_type_index2:
                    temp_checkpoint = checkpoint_list[i][k]
                    checkpoint_list[i][k]   = checkpoint_list[i][k+1]
                    checkpoint_list[i][k+1] = temp_checkpoint

            elif net_type_index1 == net_type_index2:
                if plot_acc_info1['net_type'] == 'resnet':
                    # depth, bottleneck 순으로 정렬
                    if 'depth' in plot_acc_info1:
                        depth1 = plot_acc_info1['depth'][0]
                        depth2 = plot_acc_info2['depth'][0]
                        if depth1 > depth2:
                            temp_checkpoint = checkpoint_list[i][k]
                            checkpoint_list[i][k] = checkpoint_list[i][k + 1]
                            checkpoint_list[i][k + 1] = temp_checkpoint
                        elif depth1 == depth2:
                            if 'bottleneck' in plot_acc_info1:
                                bottleneck_index1 = net_info[3].index(str(plot_acc_info1['bottleneck']))
                                bottleneck_index2 = net_info[3].index(str(plot_acc_info2['bottleneck']))
                                if bottleneck_index1 > bottleneck_index2:
                                    temp_checkpoint = checkpoint_list[i][k]
                                    checkpoint_list[i][k]   = checkpoint_list[i][k+1]
                                    checkpoint_list[i][k+1] = temp_checkpoint
                        else:
                            continue

                elif plot_acc_info1['net_type'] == 'wide_resnet':
                    # depth, widen_factor 순으로 정렬
                    if 'depth' in plot_acc_info1:
                        depth1 = plot_acc_info1['depth'][0]
                        depth2 = plot_acc_info2['depth'][0]
                        if depth1 > depth2:
                            temp_checkpoint = checkpoint_list[i][k]
                            checkpoint_list[i][k] = checkpoint_list[i][k + 1]
                            checkpoint_list[i][k + 1] = temp_checkpoint
                        elif depth1 == depth2:
                            if 'widen_factor' in plot_acc_info1:
                                widen_factor1 = plot_acc_info1['widen_factor']
                                widen_factor2 = plot_acc_info2['widen_factor']

                                if widen_factor1 > widen_factor2:
                                    temp_checkpoint = checkpoint_list[i][k]
                                    checkpoint_list[i][k] = checkpoint_list[i][k + 1]
                                    checkpoint_list[i][k + 1] = temp_checkpoint
                        else:
                            continue

                elif plot_acc_info1['net_type'] == 'preact_resnet':
                    # depth, bottleneck 순으로 정렬
                    if 'depth' in plot_acc_info1:
                        depth1 = plot_acc_info1['depth'][0]
                        depth2 = plot_acc_info2['depth'][0]
                        if depth1 > depth2:
                            temp_checkpoint = checkpoint_list[i][k]
                            checkpoint_list[i][k] = checkpoint_list[i][k + 1]
                            checkpoint_list[i][k + 1] = temp_checkpoint
                        elif depth1 == depth2:
                            if 'bottleneck' in plot_acc_info1:
                                bottleneck_index1 = net_info[3].index(str(plot_acc_info1['bottleneck']))
                                bottleneck_index2 = net_info[3].index(str(plot_acc_info1['bottleneck']))
                                if bottleneck_index1 > bottleneck_index2:
                                    temp_checkpoint = checkpoint_list[i][k]
                                    checkpoint_list[i][k] = checkpoint_list[i][k + 1]
                                    checkpoint_list[i][k + 1] = temp_checkpoint
                        else:
                            continue

                elif plot_acc_info1['net_type'] == 'RefineNet_PRN':
                    # depth, bottleneck 순으로 정렬
                    if 'depth' in plot_acc_info1:
                        depth1 = plot_acc_info1['depth'][0]
                        depth2 = plot_acc_info2['depth'][0]
                        if depth1 > depth2:
                            temp_checkpoint = checkpoint_list[i][k]
                            checkpoint_list[i][k] = checkpoint_list[i][k + 1]
                            checkpoint_list[i][k + 1] = temp_checkpoint
                        elif depth1 == depth2:
                            if 'bottleneck' in plot_acc_info1:
                                bottleneck_index1 = net_info[3].index(str(plot_acc_info1['bottleneck']))
                                bottleneck_index2 = net_info[3].index(str(plot_acc_info1['bottleneck']))
                                if bottleneck_index1 > bottleneck_index2:
                                    temp_checkpoint = checkpoint_list[i][k]
                                    checkpoint_list[i][k] = checkpoint_list[i][k + 1]
                                    checkpoint_list[i][k + 1] = temp_checkpoint
                        else:
                            continue



                else:
                    continue
                # elif checkpoint_list[i][k]['net_type'] == 'preact_resnet':
            else:
                continue
            # sort_loop_end = sort_loop_end - 1

    # # ======================================================================================= # #
    # # ==================================legend text=================================== # #
    for j in range(len(checkpoint_list[i])):
        plot_acc_info1 = checkpoint_list[i][j]['plot_acc_info']
        for k, net_type in enumerate(net_info[1]):  # net_info[1] : net_type list
            if plot_acc_info1['net_type'] == net_type:
                legend_name = text_info[1][k]

                legend_sub = ''

                for l, component in enumerate(net_info[2]):  # net_info[2] : network component

                    if component in plot_acc_info1:

                        if component == 'depth':
                            legend_name = legend_name + '-' + str(plot_acc_info1[component][0])

                        else:
                            if component != 'bottleneck':
                                if legend_sub != '':
                                    legend_sub = legend_sub + ', '

                                legend_sub = legend_sub + text_info[2][l] + str(plot_acc_info1[component])

                            elif (component == 'bottleneck') and (plot_acc_info1[component] == True):
                                if legend_sub != '':
                                    legend_sub = legend_sub + ', '

                                legend_sub = legend_sub + 'bottleneck'


                            else:
                                legend_name = legend_name
                                legend_sub = legend_sub

                if legend_sub != '':
                    legend_list[i].append(legend_name + ' (' + legend_sub + ')')
                else:
                    legend_list[i].append(legend_name)


    print('check')



# ===================================================================================== #
# ====================== generate error list for plot ================================= #
top1_train_error_list = []
top5_train_error_list = []

top1_test_error_list  = []
top5_test_error_list  = []

best_top1_error_list = []
best_top5_error_list = []


def error_list(acc_list):
    acc = np.array(acc_list, dtype=np.float32)
    error = 100. - acc
    return error



for i in range(len(net_info[0])):
    top1_train_error_list.append([])
    top5_train_error_list.append([])
    top1_test_error_list.append([])
    top5_test_error_list.append([])
    best_top1_error_list.append([])
    best_top5_error_list.append([])

    for j in range(len(checkpoint_list[i])):
        summary = checkpoint_list[i][j]['summary']
        top1_train_error = error_list(summary['train_top1_acc'])
        top1_train_error_list[i].append(top1_train_error)

        top1_test_error = error_list(summary['test_top1_acc'])
        top1_test_error_list[i].append(top1_test_error)

        top5_train_error = error_list(summary['train_top5_acc'])
        top5_train_error_list[i].append(top5_train_error)

        top5_test_error = error_list(summary['test_top5_acc'])
        top5_test_error_list[i].append(top5_test_error)

        best_top1_error_list[i].append(round((100. - checkpoint_list[i][j]['best_top1_acc']), 4))
        best_top5_error_list[i].append(round((100. - checkpoint_list[i][j]['best_top5_acc']), 4))


    print('check')

print(len(top1_train_error_list))
# ============================================================================================ #
# ======================================= plotint data ======================================= #
cm = plt.cm.get_cmap('tab20')
linestyle = ['-', '--'] # '-' : test error, '--' : train error

x_axis_start = 0
x_axis_end   = 450

y_axis_start = 0
y_axis_end   = 60

major_xticks = np.arange(x_axis_start, x_axis_end, 50)
x_axis_range = np.arange(x_axis_start, x_axis_end, 1)

linewidth = 1

for i in range(len(net_info[0])): # len(top1_train_error_list) == len(net_info[0])
    fig = plt.figure(figsize=(20, 12))
    # ax1 = plt.subplot()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title(text_info[0][i], fontsize = 14, fontweight = 'bold')

    ax1.set_xticks(major_xticks)
    ax1.grid(True)
    ax1.axis([x_axis_start, x_axis_end, y_axis_start, y_axis_end])

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(text_info[3][0])

    for j in range(len(checkpoint_list[i])):
        col_idx = j % len(cm.colors) * 2
        label = None
        print(len(top1_test_error_list[i][j]))
        axis_range = np.arange(0, len(top1_test_error_list[i][j]), 1)

        ax1.plot(axis_range, top1_test_error_list[i][j], color=cm.colors[col_idx],
                linestyle=linestyle[0], linewidth=linewidth, label=legend_list[i][j])
        ax1.plot(axis_range, top1_train_error_list[i][j], color=cm.colors[col_idx],
                linestyle=linestyle[1], linewidth=linewidth, label=None)

    plt.legend() # ============위치 중요 ==============#


    ax2 = fig.add_subplot(2,1,2)
    best_error_list = []
    best_error_list.append(best_top1_error_list[i])
    best_error_list.append(best_top5_error_list[i])

    table = plt.table(cellText=best_error_list,
                      colWidths=[0.1]*len(legend_list[i]),
                      rowLabels=text_info[3],
                      colLabels=legend_list[i],
                      loc='best')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    ax2.axis('off')

    plt.show()
    print('check')


















