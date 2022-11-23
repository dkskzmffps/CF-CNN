from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import JH_utile
import JH_net_param_setting as NetSet
import torchvision.datasets as datasets
from pytorchcv.model_provider import get_model as ptcv_get_model

import math
import time
import torch.optim as optim





import argparse



net_type = 'pyramid_net'
# resnet, wide_resnet, preact_resnet, pyramid_net, pyramid_net_SD (SD: Shake Drop)

# args = NetSet.net_param_setting(net_type)
#
# # Display network settings.
# JH_utile.print_Net_Setting(args)



net_info = { 0: ['imagenet'],                    # dataset
             # 1: ['resnet', 'wide_resnet', 'preact_resnet', 'pyramid_net'], # net_type
             # 2: ['depth', 'widen_factor', 'alpha', 'bottleneck'],    # net_component
             # 3: ['False', 'True']                           # bool variable list
           }

args = NetSet.net_param_setting(net_type, net_info[0][0])

# Display network settings.
JH_utile.print_Net_Setting(args)


num_workers = 8


base_save_path   = '/home/jinho/0_project_temp/'
dataset_path     = base_save_path + 'data/'
checkpoint_path  = base_save_path + 'checkpoint/'
class_score_path = base_save_path + 'class_score/'


for i, dataset in enumerate(net_info[0]):
    class_score_path_list = ['class_score',
                             dataset]
    _ = JH_utile.make_Dir(base_save_path, class_score_path_list)





# #=====================================================================================================================#
# # Checkpoint list load ===============================================================================================#
#
# checkpoint_filename_list = []
# checkpoint_filename = []
# # checkpoint_list=[]
#
# for i, dataset in enumerate(net_info[0]):
#     checkpoint_filename_list.append([])
#     # checkpoint_list.append([])
#     checkpoint_file_path = checkpoint_path + dataset + os.sep
#
#     assert os.path.isdir(checkpoint_file_path), \
#         'Error: No checkpoint exist.'
#
#     checkpoint_list_temp = os.listdir(checkpoint_file_path)
#     checkpoint_list_temp.sort()
#
#     for j in range(len(checkpoint_list_temp)):
#         if 'best' in checkpoint_list_temp[j]:
#             checkpoint_filename_list[i].append(checkpoint_list_temp[j])
#             # checkpoint = torch.load(checkpoint_file_path + checkpoint_list_temp[j])
#             # checkpoint_list[i].append(checkpoint)
#
#         else:
#             continue


#=====================================================================================================================#
# Device setting =====================================================================================================#

use_cuda        = torch.cuda.is_available()
# use_cuda = False
# network device define
if use_cuda:
    device = torch.device('cuda:0')
    device_location = 'cuda:0'
    data_parallel_device = [0]
    # data_parallel_device = range(torch.cuda.device_count())
else:
    device = torch.device('cpu')
    device_location = 'cpu'



for i, dataset in enumerate(net_info[0]):
    # checkpoint_file_path = checkpoint_path + dataset + os.sep
    class_score_file_path = class_score_path + dataset + os.sep
    # assert os.path.isdir(checkpoint_file_path), 'Error: No Checkpoint Dir. exist.'
    assert os.path.isdir(class_score_file_path), 'Error: No Class score Dir. exist.'

    # data load, calculate mean and std
    parser = argparse.ArgumentParser(description='temp_dataset')
    parser.add_argument('--dataset', default=dataset, type=str)
    args_dataset = parser.parse_args()

    print(args_dataset.dataset)


    if dataset.startswith('cifar'):
        train_data, train_labels, test_data, test_labels, num_classes = JH_utile.data_load(dataset_path, args_dataset)
        num_classes = [num_classes]

        data_mean, data_std = JH_utile.meanstd(train_data)
        data_mean = tuple(data_mean / 255.0)  # convert numpy array to tuple
        data_std = tuple(data_std / 255.0)

        batch_size = 400

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ])  # meanstd transformation

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ])



        testset = JH_utile.MyImageDataSet(test_data, test_labels, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        trainset2 = JH_utile.MyImageDataSet(train_data, train_labels, transform=transform_test)
        trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=batch_size, shuffle=False, num_workers=num_workers)



    elif dataset == 'imagenet':
        traindir = os.path.join(dataset_path, 'train')
        valdir = os.path.join(dataset_path, 'val')
        batch_size = 150
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        traindir = dataset_path + '/imagenet/train'
        valdir = dataset_path + '/imagenet/val'

        trainset2 = datasets.ImageFolder(traindir, transform=transform_test)
        testset = datasets.ImageFolder(valdir, transform=transform_test)


        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        num_classes = [len(trainset2.classes)]

    # for j in range(len(checkpoint_filename_list[i])):
    # saved_filename = checkpoint_file_path + checkpoint_filename_list[i][j]
    # checkpoint = torch.load(saved_filename, map_location=device_location)

    # args = NetSet.net_param_setting(checkpoint['args'].net_type)
    # args = checkpoint['args']

    # args, state_dict, best_top1_acc, best_top5_acc, summary = JH_utile.load_param(args, checkpoint)

    file_name = JH_utile.get_filename(args, num_classes)

    net = ptcv_get_model("efficientnet_b4c", pretrained=True)
    a = net.features.stage1
    # net = ptcv_get_model("senet16", pretrained=True)
    # net.load_state_dict(state_dict)

    if use_cuda:
        net.to(device)
        net = nn.DataParallel(net, device_ids=data_parallel_device)
        cudnn.benchmark = True


    # Estimate class score and save
    train_set_class_score, train_set_label = JH_utile.save_class_score(net, trainloader2, len(trainset2), device)
    # test_set_class_score, test_set_label = JH_utile.save_class_score(net, testloader, len(testset), device)
    # train_set_class_score, train_set_label = JH_utile.save_class_score_without_softmax(net, trainloader2, len(trainset2), device)
    # test_set_class_score, test_set_label = JH_utile.save_class_score_without_softmax(net, testloader, len(testset), device)

    # state = {
    #         'test_set_class_score'      : None,
    #         'test_set_label'            : None,
    #         'train_set_class_score'     : train_set_class_score,
    #         'train_set_label'           : train_set_label,
    #         'num_classes'               : num_classes,
    #
    #         'test_set_class_score_np'   : None,
    #         'train_set_class_score_np'  : train_set_class_score.numpy(),
    #         'checkpoint_filename'       : 'pretrained_pyramidnet101_a360_from_torchcv',
    #         'args'                      : args
    #         }

    # state = {
    #     'test_set_class_score': test_set_class_score,
    #     'test_set_label': test_set_label,
    #     'train_set_class_score': train_set_class_score,
    #     'train_set_label': train_set_label,
    #     'num_classes': num_classes,
    #
    #     'test_set_class_score_np': test_set_class_score.numpy(),
    #     'train_set_class_score_np': train_set_class_score.numpy(),
    #     'checkpoint_filename': 'pretrained_pyramidnet101_a360_from_torchcv',
    #     'args': args
    # }

    # class_score_save_path = class_score_file_path + file_name + '_class_score.pth'
    # torch.save(state, class_score_save_path)


    ## Group Spliting =========================================================================================
    # =========================================================================================================
    result_group_split_path_list = ['result_group_split',
                                    dataset]
    _ = JH_utile.make_Dir(base_save_path, result_group_split_path_list)
    result_group_split_path = base_save_path + 'result_group_split/' + dataset + os.sep

    num_classes = num_classes[0]

    number_group_target_list = [1, 100, 5]
    h_level = len(number_group_target_list)  # hierarchical level
    lamda = [1, 1, 5]

    split_target = 'train'
    num_epochs_split = 15000

    ori_label = np.arange(num_classes)
    ori_label = ori_label.tolist()
    tree_history_init = []
    for i in range(num_classes):
        tree_history_init.append(0)

    group_label_info = {'h_level': h_level,
                        'number_group_target_list': number_group_target_list,
                        # target number of split in each hierarchical level
                        'lamda': lamda,
                        # 'tree_history'         : [tree_history_init],
                        'group_label': [ori_label],  # h_level * num_classes
                        'num_group': [1],  # number of group in each hierarchical level
                        'num_classes_in_group': [[num_classes]],
                        # number of class in each group at each hierarchical level
                        'class_list_in_group': [[ori_label]],  # h_level * num_group * class_list
                        'args': args
                        }

    lamda1 = lamda[0]
    lamda2 = lamda[1]
    lamda3 = math.pow(0.1, lamda[2])
    lamda3 = round(lamda3, lamda[2])

    num_train_set = train_set_class_score.size()[0]
    num_each_class_train = num_train_set / num_classes
    # num_test_set = test_set_class_score.size()[0]
    # num_each_class_test = num_test_set / num_classes

    if split_target == 'train':
        num_data = num_train_set
        data_class_score = train_set_class_score
        data_label = train_set_label
    # elif split_target == 'test':
    #     num_data = num_test_set
    #     data_class_score = test_set_class_score
    #     data_label = test_set_label
    else:
        assert split_target == 'train' or split_target == 'test', 'Error: Target is not defined!'

    data_label_list = []
    for i in range(len(data_label)):
        temp_data_label = data_label[i]
        data_label_list.append(temp_data_label.data.item())

    data_label = data_label_list

    start_time = time.time()
    for h in range(h_level - 1):

        num_g_accu = 0
        temp_group_label = torch.zeros(num_classes, dtype=torch.int64)
        # temp_num_group            = []
        temp_num_classes_in_group = []
        temp_class_list_in_group = []

        num_classes_split_in_group = []
        class_list_split_in_group = []

        for g in range(group_label_info['num_group'][h]):

            # data set setting
            if split_target == 'train':
                num_data = num_train_set
            # elif split_target == 'test':
            #     num_data = num_test_set
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

                        mean_score_data[n_c] += data_class_score[n_d]
                        count_each_class[n_c] = count_each_class[n_c] + 1

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
                disjoint_loss = split_module(mean_score_data)
                overlap_loss = split_module.overlap_loss()
                balance_loss = split_module.balance_loss()
                total_loss = lamda1 * disjoint_loss + lamda2 * overlap_loss + lamda3 * balance_loss

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
                disjoint_loss_after_split = split_module(mean_score_data, p_replace)
                overlap_loss_after_split = split_module.overlap_loss(p_replace)
                balance_loss_after_split = split_module.balance_loss(p_replace)

            print('check')

            num_g_accu += len(num_classes_split_in_group)

        temp_num_group = len(temp_num_classes_in_group)
        group_label_info['num_group'].append(temp_num_group)
        group_label_info['num_classes_in_group'].append(temp_num_classes_in_group)
        group_label_info['class_list_in_group'].append(temp_class_list_in_group)

        for n_g in range(temp_num_group):
            for n_c_g in range(len(temp_class_list_in_group[n_g])):
                group_label_idx = temp_class_list_in_group[n_g][n_c_g]
                temp_group_label[group_label_idx] = n_g

        group_label_info['group_label'].append(temp_group_label.tolist())

    assert args.dataset == 'cifar100' or args.dataset == 'cifar10' or args.dataset == 'imagenet', 'Check: dataset'
    if args.dataset == 'cifar100':
        num_classes = [100]
    elif args.dataset == 'cifar10':
        num_classes = [10]
    elif args.dataset == 'imagenet':
        num_classes = [1000]
    file_name = JH_utile.get_filename_for_split(args, group_label_info, num_classes)

    torch.save(group_label_info, result_group_split_path + file_name + '.pth')

    epoch_time = time.time() - start_time

    print(epoch_time)
    print('check')
