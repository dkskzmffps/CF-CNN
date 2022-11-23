# -*- coding: utf-8 -*-

# from __future__ import print_function
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
# import os
# import time
# import argparse
# from networks import wide_resnet   as WRN
# import JH_utile
# import sys


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import argparse
from networks import wide_resnet   as WRN
from networks import resnet        as RN
from networks import preact_resnet as PRN
from networks import pyramid_net   as PyN
# from networks import preact_resnet_test as PRNT
import JH_utile
import JH_net_param_setting as NetSet
import sys
import math
import os
import time


net_info = { 0: ['cifar10', 'cifar100'],                    # dataset
             1: ['resnet', 'wide_resnet', 'preact_resnet', 'pyramid_net'], # net_type
             2: ['depth', 'widen_factor', 'alpha', 'bottleneck'],    # net_component
             3: ['False', 'True']                           # bool variable list
           }


# Path for save checkpoint
base_save_path              = '/home/jinho/0_project_temp/'
dataset_path = '/home/jinho/0_project_temp/data/'
checkpoint_path             = base_save_path + 'checkpoint/' + net_info[0][1] + os.sep
result_group_split_path     = base_save_path + 'result_group_split/' + net_info[0][1] + os.sep
class_score_path            = base_save_path + 'class_score/' + net_info[0][1] + os.sep
# Group Split Information load
group_label_info_filename = 'wide_resnet-28x12_h3_G-01-02-04_S-01-02-02_L-01-01-(10e-5)'
group_label_info = torch.load(result_group_split_path + group_label_info_filename + '.pth')
print("| Group label information are loaded")

args = group_label_info['args']

# option
args.start_epoch = 1
args.lr = 0.05
args.resume             = True
args.testOnly           = False
args.save_class_score   = False


h_level = group_label_info['h_level']

# Build network & define file name
print('\n[Phase 2] : Model setup')

file_name = JH_utile.get_filename_for_split(args, group_label_info)

# # Hyper Parameter settings
# use_cuda        = torch.cuda.is_available()
# best_acc        = 0
# best_state_dict = None
# test_batch_size = 400
# optim_type      = 'SGD'

#=====================================================================================================================#
# settings variable ==================================================================================================#
best_top1_acc   = 0
best_top5_acc   = 0
best_state_dict = None
optim_type      = 'SGD'
summary         = {
                    'train_loss'     :[],
                    'train_top1_acc' :[],
                    'train_top5_acc' :[],
                    'test_loss'      :[],
                    'test_top1_acc'  :[],
                    'test_top5_acc'  :[]
                    }
checkpoint_interval = 10


use_cuda        = torch.cuda.is_available()
# network device define
if use_cuda:
    device = torch.device('cuda:0')
    device_location = 'cuda:0'
    # data_parallel_device = [0]
    data_parallel_device = range(torch.cuda.device_count())
else:
    device = torch.device('cpu')
    device_location = 'cpu'



# data load, calculate mean and std
train_data, train_labels, test_data, test_labels, num_classes = JH_utile.data_load(dataset_path, args)


data_mean, data_std = JH_utile.meanstd(train_data)
data_mean           = tuple(data_mean/255.0)  # convert numpy array to tuple
data_std            = tuple(data_std/255.0)


# Data preprocessing setting and preparation
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(data_mean, data_std),
                                     ]) # meanstd transformation


transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(data_mean, data_std),
                                    ])


# Dataset preparation.
trainset    = JH_utile.MyImageDataSet(train_data, train_labels, transform=transform_train)
testset     = JH_utile.MyImageDataSet(test_data, test_labels, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testloader  = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
# Dataset for save score
trainset2    = JH_utile.MyImageDataSet(train_data, train_labels, transform=transform_test)
trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=args.batch_size, shuffle=False, num_workers=8)


print("| Data set are ready.")






# =================== Num Classes setting ================#
#      추후 수정 필요 #
num_group = group_label_info['num_group']  # num_group[0]이 전체 클레스 개수가 저장되도록 수정 필요

if h_level == 2:
    num_classes = [num_classes, num_group[1]]
elif h_level == 3:
    num_classes = [num_classes, num_group[1], num_group[2]]
elif h_level == 4:
    num_classes = [num_classes, num_group[1], num_group[2], num_group[3]]
else:
    assert h_level == 2 or h_level == 3 or h_level == 4, 'Error: plese check h_level and num_group'







if args.resume or args.testOnly or args.save_class_score:
    # Load checkpoint
    assert os.path.isdir(checkpoint_path), 'Error: No checkpoint directory found!'

    # # 클래스 (각 그룹의) 정보 추가 되어야 함.
    if args.resume:
        saved_filename = checkpoint_path + file_name + '.pth'
    elif args.testOnly or args.save_class_score:
        saved_filename = checkpoint_path + file_name + 'best.pth'

    checkpoint = torch.load(saved_filename, map_location=device_location)

    args, state_dict, best_top1_acc, best_top5_acc, summary, _ = JH_utile.load_param(args, checkpoint)


    # Build network & define file name
    net = JH_utile.get_network_for_split(args, num_classes)
    net.load_state_dict(state_dict)
    print('a')
else:
    print('| Building net type [' + args.net_type + ']...')
    net = JH_utile.get_network_for_split(args, num_classes)


    # weight initialization
    if args.net_type == 'wide_resnet':
        net.apply(WRN.weight_init)
    elif args.net_type == 'resnet':
        net.apply(RN.weight_init)
    # elif args.net_type == 'preact_resnet':
    #     net.apply(PRNT.weight_init)
    elif args.net_type == 'preact_resnet':
        net.apply(PRN.weight_init)
    elif args.net_type == 'pyramid_net':
        net.apply(PyN.weight_init)

if use_cuda:
    net.to(device)
    net = nn.DataParallel(net, device_ids=data_parallel_device)
    cudnn.benchmark = True

# only save class score
if (args.save_class_score):
    # state = save_class_score(args, net, trainloader2, testloader, len(trainset2), len(testset), device)
    train_set_class_score, train_set_label  = JH_utile.save_class_score(net, trainloader2, len(trainset2), device)
    test_set_class_score, test_set_label    = JH_utile.save_class_score(net, testloader, len(testset), device)


    state = {
        'test_set_class_score'      : test_set_class_score,
        'test_set_label'            : test_set_label,
        'train_set_class_score'     : train_set_class_score,
        'train_set_label'           : train_set_label,
        'num_classes'               : num_classes,

        'test_set_class_score_np'   : test_set_class_score.numpy(),
        'train_set_class_score_np'  : train_set_class_score.numpy(),
    }

    torch.save(state, class_score_path + file_name + '_class_score.pth')

    sys.exit(0)

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')

    net.eval()

    top1_acc    = JH_utile.AverageMeter()
    top5_acc    = JH_utile.AverageMeter()


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs, targets, topk_range=(1, args.topk))

            top1_acc.update(top1_acc_temp, inputs.size(0))
            top5_acc.update(top5_acc_temp, inputs.size(0))

        print("| Test Result\tAcc@1: %.8f%%" % (top1_acc.avg))

        sys.exit(0)



criterion = nn.CrossEntropyLoss()

group_label = group_label_info['group_label']

# record_acc = {'train_acc'   : [],
#               'train_error' : [],
#               'test_acc'    : [],
#               'test_error'  : []
#               }

# Training
def train(epoch):
    global net
    global best_state_dict

    # # learning rate 다운시 최적의 파라미터값 로드
    # for i in range(len(args.lr_drop_epoch)):
    #     if epoch == (args.lr_drop_epoch[i]+1):
    #         # network 다시 빌드 : nn.DataParallel 이 적용된 네트워크에 파라미터 로드가 안됨
    #         net = WRN.get_network(args, num_classes)
    #         # net.load_state_dict(best_state_dict)
    #         net.load_state_dict(best_state_dict)
    #         if use_cuda:
    #             # net.cuda()
    #             net.to(device)
    #             net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #             cudnn.benchmark = True


    # global best_state_dict
    net.train()
    train_loss = JH_utile.AverageMeter()
    top1_acc = JH_utile.AverageMeter()
    top5_acc = JH_utile.AverageMeter()

    lr = JH_utile.learning_rate(args, epoch)  # learning_rate
    mmt, w_decay = JH_utile.momentum_weightdecay(args)  # momentum and weight decay
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mmt, weight_decay=w_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))
    for batch_idx, (inputs, targets1) in enumerate(trainloader):

        # ========================================================================= #
        # =======================  H Level 2======================================= #
        if h_level == 2:
            # temp =group_label_info

            temp_targets2 = []

            for i in range(len(inputs)):
                temp_targets2.append(group_label[1][targets1[i]])

            targets2 = torch.tensor(temp_targets2)

            # GPU settings
            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)


            optimizer.zero_grad()
            outputs1, outputs2 = net(inputs)               # Forward Propagation
            loss1 = criterion(outputs1, targets1)  # Loss
            loss2 = criterion(outputs2, targets2)  # Loss

            ## Disjoint loss calculate
            if (args.disjoint == True):

                disjoint_loss1 = JH_utile.Disjoint_loss(outputs1)
                disjoint_loss2 = JH_utile.Disjoint_loss(outputs2)
                total_disjoint_loss = disjoint_loss1 + disjoint_loss2

                total_loss = loss1 + loss2 + total_disjoint_loss

            else:
                total_loss = loss1 + loss2

            top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs1, targets1, topk_range=(1, args.topk))

            train_loss.update(total_loss.data.item(), inputs.size(0))
            top1_acc.update(top1_acc_temp, inputs.size(0))
            top5_acc.update(top5_acc_temp, inputs.size(0))

            total_loss.backward()
            optimizer.step() # Optimizer update

        # ========================================================================= #
        # =======================  H Level 3======================================= #
        elif h_level == 3:
            # temp =group_label_info

            temp_targets2 = []
            temp_targets3 = []

            for i in range(len(inputs)):
                temp_targets2.append(group_label[1][targets1[i]])
                temp_targets3.append(group_label[2][targets1[i]])

            targets2 = torch.tensor(temp_targets2)
            targets3 = torch.tensor(temp_targets3)


            # GPU settings
            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)


            optimizer.zero_grad()
            outputs1, outputs2, outputs3 = net(inputs)               # Forward Propagation
            loss1 = criterion(outputs1, targets1)  # Loss
            loss2 = criterion(outputs2, targets2)  # Loss
            loss3 = criterion(outputs3, targets3)  # Loss

            ## Disjoint loss calculate
            if (args.disjoint == True):

                disjoint_loss1 = JH_utile.Disjoint_loss(outputs1)
                disjoint_loss2 = JH_utile.Disjoint_loss(outputs2)
                disjoint_loss3 = JH_utile.Disjoint_loss(outputs3)
                total_disjoint_loss = disjoint_loss1 + disjoint_loss2 + disjoint_loss3

                total_loss = loss1 + loss2 + loss3 + total_disjoint_loss

            else:
                total_loss = loss1 +(0.5* loss2) + (0.5*loss3)

            top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs1, targets1, topk_range=(1, args.topk))

            train_loss.update(total_loss.data.item(), inputs.size(0))
            top1_acc.update(top1_acc_temp, inputs.size(0))
            top5_acc.update(top5_acc_temp, inputs.size(0))

            total_loss.backward()
            optimizer.step() # Optimizer update

        # ========================================================================= #
        # =======================  H Level 4======================================= #
        elif h_level == 4:
            # temp =group_label_info

            temp_targets2 = []
            temp_targets3 = []
            temp_targets4 = []

            for i in range(len(inputs)):
                temp_targets2.append(group_label[1][targets1[i]])
                temp_targets3.append(group_label[2][targets1[i]])
                temp_targets4.append(group_label[3][targets1[i]])

            targets2 = torch.tensor(temp_targets2)
            targets3 = torch.tensor(temp_targets3)
            targets4 = torch.tensor(temp_targets4)


            # GPU settings
            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)
            targets4 = targets4.to(device)


            optimizer.zero_grad()
            outputs1, outputs2, outputs3, outputs4 = net(inputs)               # Forward Propagation
            loss1 = criterion(outputs1, targets1)  # Loss
            loss2 = criterion(outputs2, targets2)  # Loss
            loss3 = criterion(outputs3, targets3)  # Loss
            loss4 = criterion(outputs4, targets4)  # Loss

            ## Disjoint loss calculate
            if (args.disjoint == True):

                disjoint_loss1 = JH_utile.Disjoint_loss(outputs1)
                disjoint_loss2 = JH_utile.Disjoint_loss(outputs2)
                disjoint_loss3 = JH_utile.Disjoint_loss(outputs3)
                disjoint_loss4 = JH_utile.Disjoint_loss(outputs4)
                total_disjoint_loss = disjoint_loss1 + disjoint_loss2 + disjoint_loss3 + disjoint_loss4

                total_loss = loss1 + loss2 + loss3 + loss4 + total_disjoint_loss

            else:
                total_loss = loss1 + loss2 + loss3 + loss4

            top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs1, targets1, topk_range=(1, args.topk))

            train_loss.update(total_loss.data.item(), inputs.size(0))
            top1_acc.update(top1_acc_temp, inputs.size(0))
            top5_acc.update(top5_acc_temp, inputs.size(0))

            total_loss.backward()
            optimizer.step()  # Optimizer update


        else:
            assert h_level == 2 or h_level == 3 or h_level == 4, 'Check: please check the h_level'



        if (args.disjoint == True):

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [{CE:3d}/{TE:3d}] Iter[{CI:3d}/{TI:3d}]'
                             .format(CE=epoch, TE=args.num_epochs, CI=batch_idx + 1,
                                     TI=math.ceil(len(trainset) / trainloader.batch_size)))
            sys.stdout.write('\t| Acc: {Acc:.8f}%\tBest Acc(Val.): {BAcc}%\t Disjointloss: {DL:.8f}'
                             .format(Acc=top1_acc.val, BAcc=best_top1_acc, DL=total_disjoint_loss.data.item()))


        else:
            sys.stdout.write('\r')  # 커서를 콘솔창 맨 왼쪽에 위치 시키기
            sys.stdout.write('| Epoch [{CE:3d}/{TE:3d}] Iter[{CI:3d}/{TI:3d}]'
                             .format(CE=epoch, TE=args.num_epochs, CI=batch_idx + 1,
                                     TI=math.ceil(len(trainset) / trainloader.batch_size)))
            sys.stdout.write('\t| Acc: {Acc:.8f}%\tBest Acc(Val.): {BAcc}%'
                             .format(Acc=top1_acc.val, BAcc=best_top1_acc))
            sys.stdout.flush()
    sys.stdout.write('\n')



    summary['train_loss'].append(train_loss.avg)
    summary['train_top1_acc'].append(top1_acc.avg)
    summary['train_top5_acc'].append(top5_acc.avg)




def test(epoch):
    global best_state_dict
    global best_top1_acc
    global best_top5_acc

    net.eval()

    test_loss = JH_utile.AverageMeter()
    top1_acc = JH_utile.AverageMeter()
    top5_acc = JH_utile.AverageMeter()


    with torch.no_grad():
        for batch_idx, (inputs, targets1) in enumerate(testloader):

            # ========================================================================= #
            # =======================  H Level 2======================================= #
            if h_level == 2:
                temp_targets2 = []

                for i in range(len(inputs)):
                    temp_targets2.append(group_label[1][targets1[i]])

                targets2 = torch.tensor(temp_targets2)

                # GPU settings
                inputs = inputs.to(device)
                targets1 = targets1.to(device)
                targets2 = targets2.to(device)

                outputs1, outputs2 = net(inputs)  # Forward Propagation
                loss1 = criterion(outputs1, targets1)  # Loss
                loss2 = criterion(outputs2, targets2)  # Loss

                ## Disjoint loss calculate
                if (args.disjoint == True):

                    disjoint_loss1 = JH_utile.Disjoint_loss(outputs1)
                    disjoint_loss2 = JH_utile.Disjoint_loss(outputs2)
                    total_disjoint_loss = disjoint_loss1 + disjoint_loss2

                    total_loss = loss1 + loss2 + total_disjoint_loss


                else:
                    total_loss = loss1 + loss2

                top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs1, targets1, topk_range=(1, args.topk))

                test_loss.update(total_loss.data.item(), inputs.size(0))
                top1_acc.update(top1_acc_temp, inputs.size(0))
                top5_acc.update(top5_acc_temp, inputs.size(0))

                sys.stdout.write('\r')
                sys.stdout.write('| Validation Iter[{CI:3d}/{TI:3d}]'
                                 .format(CI=batch_idx + 1, TI=math.ceil(len(testset) / testloader.batch_size)))
                sys.stdout.flush()

            # ========================================================================= #
            # =======================  H Level 3======================================= #
            elif h_level == 3:
                temp_targets2 = []
                temp_targets3 = []

                for i in range(len(inputs)):
                    temp_targets2.append(group_label[1][targets1[i]])
                    temp_targets3.append(group_label[2][targets1[i]])

                targets2 = torch.tensor(temp_targets2)
                targets3 = torch.tensor(temp_targets3)

                # GPU settings
                inputs = inputs.to(device)
                targets1 = targets1.to(device)
                targets2 = targets2.to(device)
                targets3 = targets3.to(device)

                outputs1, outputs2, outputs3 = net(inputs)  # Forward Propagation
                loss1 = criterion(outputs1, targets1)  # Loss
                loss2 = criterion(outputs2, targets2)  # Loss
                loss3 = criterion(outputs3, targets3)  # Loss



                ## Disjoint loss calculate
                if (args.disjoint == True):

                    disjoint_loss1 = JH_utile.Disjoint_loss(outputs1)
                    disjoint_loss2 = JH_utile.Disjoint_loss(outputs2)
                    disjoint_loss3 = JH_utile.Disjoint_loss(outputs3)
                    total_disjoint_loss = disjoint_loss1 + disjoint_loss2 + disjoint_loss3

                    total_loss = loss1 + loss2 + loss3 + total_disjoint_loss


                else:
                    total_loss = loss1 + loss2 + loss3

                top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs1, targets1, topk_range=(1, args.topk))

                test_loss.update(total_loss.data.item(), inputs.size(0))
                top1_acc.update(top1_acc_temp, inputs.size(0))
                top5_acc.update(top5_acc_temp, inputs.size(0))

                sys.stdout.write('\r')
                sys.stdout.write('| Validation Iter[{CI:3d}/{TI:3d}]'
                                 .format(CI=batch_idx + 1, TI=math.ceil(len(testset) / testloader.batch_size)))
                sys.stdout.flush()

            # ========================================================================= #
            # =======================  H Level 4======================================= #
            elif h_level == 4:
                temp_targets2 = []
                temp_targets3 = []
                temp_targets4 = []

                for i in range(len(inputs)):
                    temp_targets2.append(group_label[1][targets1[i]])
                    temp_targets3.append(group_label[2][targets1[i]])
                    temp_targets4.append(group_label[3][targets1[i]])

                targets2 = torch.tensor(temp_targets2)
                targets3 = torch.tensor(temp_targets3)
                targets4 = torch.tensor(temp_targets4)

                # GPU settings
                inputs = inputs.to(device)
                targets1 = targets1.to(device)
                targets2 = targets2.to(device)
                targets3 = targets3.to(device)
                targets4 = targets4.to(device)

                outputs1, outputs2, outputs3, outputs4 = net(inputs)  # Forward Propagation
                loss1 = criterion(outputs1, targets1)  # Loss
                loss2 = criterion(outputs2, targets2)  # Loss
                loss3 = criterion(outputs3, targets3)  # Loss
                loss4 = criterion(outputs4, targets4)  # Loss



                ## Disjoint loss calculate
                if (args.disjoint == True):

                    disjoint_loss1 = JH_utile.Disjoint_loss(outputs1)
                    disjoint_loss2 = JH_utile.Disjoint_loss(outputs2)
                    disjoint_loss3 = JH_utile.Disjoint_loss(outputs3)
                    disjoint_loss4 = JH_utile.Disjoint_loss(outputs4)
                    total_disjoint_loss = disjoint_loss1 + disjoint_loss2 + disjoint_loss3 + disjoint_loss4

                    total_loss = loss1 + loss2 + loss3 + loss4 + total_disjoint_loss


                else:
                    total_loss = loss1 + loss2 + loss3 + loss4

                top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs1, targets1, topk_range=(1, args.topk))

                test_loss.update(total_loss.data.item(), inputs.size(0))
                top1_acc.update(top1_acc_temp, inputs.size(0))
                top5_acc.update(top5_acc_temp, inputs.size(0))

                sys.stdout.write('\r')
                sys.stdout.write('| Validation Iter[{CI:3d}/{TI:3d}]'
                                 .format(CI=batch_idx + 1, TI=math.ceil(len(testset) / testloader.batch_size)))
                sys.stdout.flush()

            else:
                assert h_level == 2 or h_level == 3 or h_level == 4, 'Check: please check the h_level'
        # sys.stdout.write('\r')
        # sys.stdout.write('| Validation Iter[{CI:3d}/{TI:3d}]'
        #                  .format(CE=epoch, TE=args.num_epochs, CI=batch_idx + 1,
        #                          TI=math.ceil(len(testset) / testloader.batch_size)))

        if (args.disjoint == True):
            # sys.stdout.write('\r')
            # sys.stdout.write('| Validation Iter[{CI:3d}/{TI:3d}]'
            #                  .format(CE=epoch, TE=args.num_epochs, CI=batch_idx + 1,
            #                          TI=math.ceil(len(testset) / testloader.batch_size)))
            sys.stdout.write('\t\t| Acc: {Acc:.8f}%\tBest Acc(Pre.): {BAcc}%\t Disjointloss: {DL:.8f}'
                             .format(Acc=top1_acc.avg, BAcc=best_top1_acc, DL=total_disjoint_loss.data.item()))

        else:
            # sys.stdout.write('\r')  # 커서를 콘솔창 맨 왼쪽에 위치 시키기
            # sys.stdout.write('| Validation Iter[{CI:3d}/{TI:3d}]'
            #                  .format(CE=epoch, TE=args.num_epochs, CI=batch_idx + 1,
            #                          TI=math.ceil(len(testset) / testloader.batch_size)))
            sys.stdout.write(
                '\t\t| Acc: {Acc:.8f}%\tBest Acc(Pre.): {BAcc}%\n'.format(Acc=top1_acc.avg, BAcc=best_top1_acc))
            sys.stdout.flush()

        summary['test_loss'].append(test_loss.avg)
        summary['test_top1_acc'].append(top1_acc.avg)
        summary['test_top5_acc'].append(top5_acc.avg)

# =====================================================================================================================#
# =========================================== Save Network ============================================================#
# =====================================================================================================================#

    # =================================================================================================================#
    # Save Best Model==================================================================================================#
    if top1_acc.avg > best_top1_acc:
        best_top1_acc = top1_acc.avg

        # # Saving torch.nn.DataParallel Models
        # torch.nn.DataParallel is a model wrapper that enables parallel GPU utilization.
        # To save a DataParallel model generically, save the model.module.state_dict().
        # This way, you have the flexibility to load the model any way you want to any device you want.
        # nn.DataParallel 을 사용하지 않을 경우 net.state_dict()
        # DataParallel 이 적용된 환경 그대로 저장할경우 net.state_dict() --> 불러올때도 모델이 DataParallel된 상태여야 함

        if use_cuda:
            best_state_dict = net.module.state_dict()
        else:
            best_state_dict = net.state_dict()

        print('| Saving Best model...\t\t\t| Top1 = %.8f%%' % (best_top1_acc))
        state = JH_utile.save_network(args, best_state_dict, best_top1_acc, best_top5_acc, epoch, summary)

        torch.save(state, checkpoint_path + file_name + 'best.pth')

    if top5_acc.avg > best_top5_acc:
        best_top5_acc = top5_acc.avg
        state = JH_utile.save_network(args, best_state_dict, best_top1_acc, best_top5_acc, epoch, summary)
        torch.save(state, checkpoint_path + file_name + 'best.pth')

    # =================================================================================================================#
    # Save Check Point=================================================================================================#
    if epoch % checkpoint_interval == 0 or epoch == args.num_epochs:
        if use_cuda:
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()

        print('| Saving checkpoint model...')
        state = JH_utile.save_network(args, state_dict, best_top1_acc, best_top5_acc, epoch, summary)

        torch.save(state, checkpoint_path + file_name + '.pth')


print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))


elapsed_time = 0
for epoch in range(args.start_epoch, args.num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('\n| Epoch time: %d:%02d:%02d' %(JH_utile.get_hms(epoch_time)))
    print('| Elapsed time : %d:%02d:%02d'  %(JH_utile.get_hms(elapsed_time)))


print('\n[Phase 4] : Testing model')
print('* Test results : Best Acc@1 = %.8f%%' %(best_top1_acc))
