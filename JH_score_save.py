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

import argparse

net_info = { 0: ['imagenet'],                    # dataset
             # 1: ['resnet', 'wide_resnet', 'preact_resnet', 'pyramid_net'], # net_type
             # 2: ['depth', 'widen_factor', 'alpha', 'bottleneck'],    # net_component
             # 3: ['False', 'True']                           # bool variable list
           }

num_workers = 8


base_save_path   = '/home/jinho/0_project_temp/'
dataset_path     = base_save_path + 'data/'
checkpoint_path  = base_save_path + 'checkpoint/'
class_score_path = base_save_path + 'class_score/'


for i, dataset in enumerate(net_info[0]):
    class_score_path_list = ['class_score',
                             dataset]
    _ = JH_utile.make_Dir(base_save_path, class_score_path_list)


#=====================================================================================================================#
# Checkpoint list load ===============================================================================================#

checkpoint_filename_list = []
checkpoint_filename = []
# checkpoint_list=[]

for i, dataset in enumerate(net_info[0]):
    checkpoint_filename_list.append([])

    checkpoint_file_path = checkpoint_path + dataset + os.sep

    assert os.path.isdir(checkpoint_file_path), \
        'Error: No checkpoint exist.'

    checkpoint_list_temp = os.listdir(checkpoint_file_path)
    checkpoint_list_temp.sort()

    for j in range(len(checkpoint_list_temp)):
        if 'best' in checkpoint_list_temp[j]:
            checkpoint_filename_list[i].append(checkpoint_list_temp[j])
        else:
            continue


#=====================================================================================================================#
# Device setting =====================================================================================================#

use_cuda        = torch.cuda.is_available()
# use_cuda = False
# network device define
if use_cuda:
    device = torch.device('cuda:0')
    device_location = 'cuda:0'
    # data_parallel_device = [0]
    data_parallel_device = range(torch.cuda.device_count())
else:
    device = torch.device('cpu')
    device_location = 'cpu'



for i, dataset in enumerate(net_info[0]):
    checkpoint_file_path = checkpoint_path + dataset + os.sep
    class_score_file_path = class_score_path + dataset + os.sep
    assert os.path.isdir(checkpoint_file_path), 'Error: No Checkpoint Dir. exist.'
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
        batch_size = 300
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

    # =====================================================================================================================#
    # Dataset preparation=================================================================================================#


    for j in range(len(checkpoint_filename_list[i])):
        saved_filename = checkpoint_file_path + checkpoint_filename_list[i][j]
        checkpoint = torch.load(saved_filename, map_location=device_location)

        # args = NetSet.net_param_setting(checkpoint['args'].net_type)
        args = checkpoint['args']

        args, state_dict, best_top1_acc, best_top5_acc, summary = JH_utile.load_param(args, checkpoint)

        file_name = JH_utile.get_filename(args, num_classes)

        net = JH_utile.get_network_init_weight(args, num_classes)
        net.load_state_dict(state_dict)

        if use_cuda:
            net.to(device)
            net = nn.DataParallel(net, device_ids=data_parallel_device)
            cudnn.benchmark = True


        # Estimate class score and save
        train_set_class_score, train_set_label = JH_utile.save_class_score(net, trainloader2, len(trainset2), device)
        test_set_class_score, test_set_label = JH_utile.save_class_score(net, testloader, len(testset), device)
        # train_set_class_score, train_set_label = JH_utile.save_class_score_without_softmax(net, trainloader2, len(trainset2), device)
        # test_set_class_score, test_set_label = JH_utile.save_class_score_without_softmax(net, testloader, len(testset), device)

        state = {
                'test_set_class_score'      : test_set_class_score,
                'test_set_label'            : test_set_label,
                'train_set_class_score'     : train_set_class_score,
                'train_set_label'           : train_set_label,
                'num_classes'               : num_classes,

                'test_set_class_score_np'   : test_set_class_score.numpy(),
                'train_set_class_score_np'  : train_set_class_score.numpy(),
                'checkpoint_filename'       : checkpoint_filename_list[i][j],
                'args'                      : args
                }

        class_score_save_path = class_score_file_path + file_name + '_class_score.pth'
        torch.save(state, class_score_save_path)






