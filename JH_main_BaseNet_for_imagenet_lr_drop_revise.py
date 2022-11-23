# -*- coding: utf-8 -*-

# from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse

import JH_utile
import JH_DataAug
import JH_net_param_setting as NetSet
import sys
import math
import os
import time


net_type = 'preact_resnet'
target_data = 'cifar100' # cifar10, cifar100, imagenet
# resnet, wide_resnet, preact_resnet, pyramid_net, pyramid_net_SD (SD: Shake Drop)


lr_drop_revise = True
if lr_drop_revise == True:
    stored_at_backup_root = False
    load_epoch = 1 # 원하는 값으로 수정 it will be change start epoch
print('lr_drop_revise\t\t: {ldr}'.format(ldr=lr_drop_revise))
print('\n')


args = NetSet.net_param_setting(net_type, target_data)

# Display network settings.
JH_utile.print_Net_Setting(args)


#=====================================================================================================================#
#======================================== Preparing for Data Set =====================================================#
#=====================================================================================================================#


# ====================================================================================================================#
# Define path ========================================================================================================#

base_dataset_path = '/home/jinho/0_project_temp/data/'
dataset_path = os.path.join(base_dataset_path, args.dataset)

base_save_path = '/home/jinho/0_project_temp/'

checkpoint_path_list    = ['checkpoint2',
                           args.dataset,]
class_score_path_list   = ['class_score',
                           args.dataset]

save_path_lr_drop_epoch = 'backup_at_lr_drop'


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
checkpoint_interval = 1

#=====================================================================================================================#
# generate folder for data saving =======================================================================================#

checkpoint_path  = JH_utile.make_Dir(base_save_path, checkpoint_path_list)
class_score_path = JH_utile.make_Dir(base_save_path, class_score_path_list)
checkpoint_path_lr_drop = JH_utile.make_Dir(checkpoint_path, [save_path_lr_drop_epoch])
    # just make folder for checkpoint at lr drop epoch

# Dataset preparation=================================================================================================#

if args.dataset == 'imagenet':


    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'val')


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    jittering = JH_DataAug.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

    lighting = JH_DataAug.Lighting(alphastd=0.1,
                                   eigval=[0.2175, 0.0188, 0.0045],
                                   eigvec=[[-0.5675, 0.7192, 0.4009],
                                           [-0.5808, -0.0045, -0.8140],
                                           [-0.5836, -0.6948, 0.4203]])

    # Data preprocessing setting and preparation
    print('\n[Phase 1] : Data Preparation')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        jittering,
        lighting,
        normalize,
    ])  # meanstd transformation
    if args.cutout:
        transform_train.transforms.append(JH_utile.Cutout(n_holes=args.n_holes, length=args.cutout_length))


    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.ImageFolder(train_dir, transform_train)
    testset = datasets.ImageFolder(test_dir, transform_test)
    num_classes = [len(trainset.classes)]




elif args.dataset.startswith('cifar'):
    train_data, train_labels, test_data, test_labels, num_classes = JH_utile.data_load(dataset_path, args)
    num_classes = [num_classes]
    data_mean, data_std = JH_utile.meanstd(train_data)
    data_mean           = tuple(data_mean/255.0)  # convert numpy array to tuple
    data_std            = tuple(data_std/255.0)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
    ])  # meanstd transformation
    if args.cutout:
        transform_train.transforms.append(JH_utile.Cutout(n_holes=args.n_holes, length=args.cutout_length))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
    ])


    trainset = JH_utile.MyImageDataSet(train_data, train_labels, transform=transform_train)
    testset = JH_utile.MyImageDataSet(test_data, test_labels, transform=transform_test)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
testloader  = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=False)



#=====================================================================================================================#
#========================================== Build Network ============================================================#
#=====================================================================================================================#



# Build network & define file name
print('\n[Phase 2] : Model setup')

file_name = JH_utile.get_filename(args, num_classes)


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


#=====================================================================================================================#
# Build Network ======================================================================================================#

if args.resume or args.testOnly or args.save_class_score:
    # Load checkpoint
    assert os.path.isdir(checkpoint_path), 'Error: No checkpoint directory found!'

    # # 클래스 (각 그룹의) 정보 추가 되어야 함.
    if args.resume:
        if lr_drop_revise == True and stored_at_backup_root == True:
            saved_filename = checkpoint_path_lr_drop + file_name + '_' + str(load_epoch) + '.pth'
        else:
            saved_filename = checkpoint_path + file_name + '.pth'
    elif args.testOnly or args.save_class_score:
        saved_filename = checkpoint_path + file_name + '_best.pth'


    checkpoint = torch.load(saved_filename, map_location=device_location)



    if lr_drop_revise == True:
        JH_utile.network_hyperparameter_check(args, checkpoint['args'], lr_drop_revise)
        lr_drop_epoch = args.lr_drop_epoch
        args, state_dict, best_top1_acc, best_top5_acc, summary = JH_utile.load_param(args, checkpoint)
        args.lr_drop_epoch = lr_drop_epoch
        args.start_epoch = load_epoch+1
    else:
        JH_utile.network_hyperparameter_check(args, checkpoint['args'])
        args, state_dict, best_top1_acc, best_top5_acc, summary = JH_utile.load_param(args, checkpoint)


    # if args.dataset.startswith('cifar'):
    # # train set setting
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(data_mean, data_std),
    #     ])  # meanstd transformation
    #     if hasattr(args, 'cutout'):
    #         if args.cutout:
    #             transform_train.transforms.append(JH_utile.Cutout(n_holes=args.n_holes, length=args.cutout_length))
    #         # trainset = JH_utile.MyImageDataSet(train_data, train_labels, transform=transform_train)
    # elif args.dataset=='imagenet':
    #     transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])  # meanstd transformation
    #     if hasattr(args, 'cutout'):
    #         if args.cutout:
    #             transform_train.transforms.append(JH_utile.Cutout(n_holes=args.n_holes, length=args.cutout_length))



    # Build network & define file name
    net = JH_utile.get_network_init_weight(args, num_classes)
    net.load_state_dict(state_dict)
else:
    print('| Building net type [' + args.net_type + ']...')
    net = JH_utile.get_network_init_weight(args, num_classes)


if use_cuda:
    net.to(device)
    net = nn.DataParallel(net, device_ids=data_parallel_device)
    cudnn.benchmark = True


# only save class score
if (args.save_class_score):
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # # state = save_class_score(args, net, trainloader2, testloader, len(trainset2), len(testset), device)

    if args.dataset.startswith('cifar'):
        train_set_class_score, train_set_label = JH_utile.save_class_score(net, trainloader, len(trainset), device)
        test_set_class_score, test_set_label = JH_utile.save_class_score(net, testloader, len(testset), device)
    elif args.dataset == 'imagenet':
        train_set_class_score, train_set_label = JH_utile.save_class_score(net, trainloader, len(trainset.imgs), device)
        test_set_class_score, test_set_label = JH_utile.save_class_score(net, testloader, len(testset.imgs), device)




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


    print("| Test Result\tAcc@1: %.8f%%" %(top1_acc.avg))

    sys.exit(0)

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch):
    start_time = time.time()

    net.train()
    train_loss  = JH_utile.AverageMeter()
    top1_acc    = JH_utile.AverageMeter()
    top5_acc    = JH_utile.AverageMeter()


    lr = JH_utile.learning_rate(args, epoch) # learning_rate
    print(lr)
    mmt, w_decay = JH_utile.momentum_weightdecay(args) # momentum and weight decay
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mmt, weight_decay=w_decay, nesterov=args.nesterov)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, lr))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)  # GPU settings
        optimizer.zero_grad()
        outputs = net(inputs)               # Forward Propagation

        loss = criterion(outputs, targets)  # Loss

        total_loss = loss


        top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs, targets, topk_range=(1, args.topk))


        train_loss.update(total_loss.data.item(), inputs.size(0))
        top1_acc.update(top1_acc_temp, inputs.size(0))
        top5_acc.update(top5_acc_temp, inputs.size(0))

        # back_time_start = time.time()
        # network parameters update
        total_loss.backward()
        optimizer.step()  # Optimizer update
        # back_time_end = time.time() - back_time_start
        # print('| backward time: %d:%02d:%02d' % (JH_utile.get_hms(back_time_end)))

        if batch_idx % 1000 == 0:
            epoch_time = time.time() - start_time

            print('| Epoch time: %d:%02d:%02d' % (JH_utile.get_hms(epoch_time)))


        sys.stdout.write('\r') # 커서를 콘솔창 맨 왼쪽에 위치 시키기
        sys.stdout.write('| Epoch [{CE:3d}/{TE:3d}] Iter[{CI:3d}/{TI:3d}]'
                         .format(CE=epoch, TE=args.num_epochs, CI=batch_idx + 1,
                                TI=math.ceil(len(trainset) / trainloader.batch_size)))
        sys.stdout.write('\t| Acc: {Acc:.8f}%\tBest Acc(Val.): {BAcc}%'.format(Acc=top1_acc.avg, BAcc=best_top1_acc))
        sys.stdout.flush()


    sys.stdout.write('\n')


    summary['train_loss'    ].append(train_loss.avg)
    summary['train_top1_acc'].append(top1_acc.avg)
    summary['train_top5_acc'].append(top5_acc.avg)



def test(epoch):
    global best_state_dict
    global best_top1_acc
    global best_top5_acc
    # global best_epoch

    net.eval()

    test_loss   = JH_utile.AverageMeter()
    top1_acc    = JH_utile.AverageMeter()
    top5_acc    = JH_utile.AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # if use_cuda:
            #     inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets     = inputs.to(device), targets.to(device)  # GPU settings
            outputs             = net(inputs)
            loss                = criterion(outputs, targets)


            total_loss = loss

            top1_acc_temp, top5_acc_temp = JH_utile.topk_accurcy(outputs, targets, topk_range=(1, args.topk))

            test_loss.update(total_loss.data.item(), inputs.size(0))
            top1_acc.update(top1_acc_temp, inputs.size(0))
            top5_acc.update(top5_acc_temp, inputs.size(0))




            sys.stdout.write('\r')
            sys.stdout.write('| Validation Iter[{CI:3d}/{TI:3d}]'
                             .format(CI=batch_idx + 1, TI=math.ceil(len(testset) / testloader.batch_size)))
            sys.stdout.flush()

    sys.stdout.write('\r')  # 커서를 콘솔창 맨 왼쪽에 위치 시키기
    sys.stdout.write('| Validation Iter[{CI:3d}/{TI:3d}]'
                     .format(CE=epoch, TE=args.num_epochs, CI=batch_idx + 1,
                             TI=math.ceil(len(testset) / testloader.batch_size)))
    sys.stdout.write('\t\t| Acc: {Acc:.8f}%\tBest Acc(Pre.): {BAcc}%'.format(Acc=top1_acc.avg, BAcc=best_top1_acc))
    sys.stdout.flush()

    summary['test_loss'    ].append(test_loss.avg)
    summary['test_top1_acc'].append(top1_acc.avg)
    summary['test_top5_acc'].append(top5_acc.avg)


# =====================================================================================================================#
# =========================================== Save Network ============================================================#
# =====================================================================================================================#

    # =================================================================================================================#
    # Save Best Model==================================================================================================#
    if top1_acc.avg > best_top1_acc:
        best_top1_acc =top1_acc.avg

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

        print('\n| Saving Best model...\t\t\t| Top1 = %.8f%%' %(best_top1_acc))
        state = JH_utile.save_network(args, best_state_dict, best_top1_acc, best_top5_acc,
                                      epoch, summary, group_label_info=None)

        torch.save(state, checkpoint_path + file_name + '_best.pth')

    if top5_acc.avg > best_top5_acc:
        best_top5_acc = top5_acc.avg
        state = JH_utile.save_network(args, best_state_dict, best_top1_acc, best_top5_acc,
                                      epoch, summary, group_label_info=None)
        torch.save(state, checkpoint_path + file_name + '_best.pth')


    # =================================================================================================================#
    # Save Check Point=================================================================================================#
    if epoch % checkpoint_interval == 0 or epoch == args.num_epochs:
        if use_cuda:
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()

        print('| Saving checkpoint model...')
        state = JH_utile.save_network(args, state_dict, best_top1_acc, best_top5_acc,
                                      epoch, summary, group_label_info=None)

        torch.save(state, checkpoint_path + file_name + '.pth')

    for i, ld_epoch in enumerate(args.lr_drop_epoch):
        if epoch == ld_epoch-1:
            state = JH_utile.save_network(args, state_dict, best_top1_acc, best_top5_acc,
                                          epoch, summary, group_label_info=None)

            torch.save(state, checkpoint_path_lr_drop + file_name + '_' + str(epoch) + '.pth')





print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))


elapsed_time = 0
for epoch in range(args.start_epoch, args.num_epochs+1):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Epoch time: %d:%02d:%02d' %(JH_utile.get_hms(epoch_time)))
    print('| Elapsed time : %d:%02d:%02d'  %(JH_utile.get_hms(elapsed_time)))



print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.8f%%' %(best_top1_acc))
