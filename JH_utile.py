# -*- coding: utf-8 -*-
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from PIL import Image


from networks import resnet                 as RN
from networks import wide_resnet            as WRN
from networks import preact_resnet          as PRN
from networks import pyramid_net            as PYN
from networks import pyramid_net_ShakeDrop  as PYN_SD

from networks import RefineNet_PreRes   as RPRN
from networks import RefineNet_Resnet   as RRN
from networks import RefineNet_WideRes  as RWRN
from networks import RefineNet_PYN      as RPYN
from networks import RefineNet_PYN_SD   as RPYN_SD




# from networks import preact_resnet_test as PRNT


import copy
import sys
import os

import math



## ======================================================##
## ============ function for Custom Dataset =============##
## ======================================================##


class MyImageDataSet(Dataset):

    def __init__(self, data, labels, transform=None, target_transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        # to return a PIL Image (normalized 0 to 1)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class MyClassScoreDataSet(Dataset):
    def __init__(self, data, labels, transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform


    def __getitem__(self, index):
        class_score, class_label = self.data[index], self.labels[index]
        return class_score, class_label


    def __len__(self):
        return len(self.data)


def cifar10_data_load(inputfile_list):

    data = []
    labels = []
    for input_file in inputfile_list:
        with open(input_file, 'rb') as fo: # fo = open(input_file, 'rb') 로 쓸수 있음
            data_dict = cPickle.load(fo, encoding='latin1')

        # data 는 numpy array 가 리스트로 있는 형태 --> concate로 묶는다.
        # labels는 각 라벨이 리스트로 있는 형태 --> list 덧셈 연산자로 더한다.
        data.append(data_dict['data'])
        labels += data_dict['labels']
        # Each row of data is an image (each element is a byte)
        # Each row of labels is an integer (the single element is an int64)

    data = np.concatenate(data, axis=0)

    data_shape = data.shape
    data = data.reshape((data_shape[0]), 3, 32, 32) # numpy 함수
    data = data.transpose((0, 2, 3, 1))             # CHW(Channel, Height, Width) convert to HWC
    return data, labels


def cifar100_data_load(inputfile_list):

    with open(inputfile_list[0], 'rb') as fo: # fo = open(input_file, 'rb') 로 쓸수 있음
        data_dict = cPickle.load(fo, encoding='latin1')

    # data 는 numpy array 가 리스트로 있는 형태 --> concate로 묶는다.
    # labels는 각 라벨이 리스트로 있는 형태 --> list 덧셈 연산자로 더한다.
    data = data_dict['data']
    coarse_labels = data_dict['coarse_labels']
    fine_labels = data_dict['fine_labels']
    # Each row of data is an image (each element is a byte)
    # Each row of labels is an integer (the single element is an int64)

    data_shape = data.shape
    data = data.reshape((data_shape[0]), 3, 32, 32) # numpy 함수
    data = data.transpose((0, 2, 3, 1))             # CHW(Channel, Height, Width) convert to HWC
    return data, coarse_labels, fine_labels


def cifar10_classlist_load(inputfile):

    with open(inputfile[0], 'rb') as fo: # fo = open(input_file, 'rb') 로 쓸수 있음
        data_dict = cPickle.load(fo, encoding='latin1')
    return data_dict['label_names']


def cifar100_classlist_load(inputfile):

    with open(inputfile[0], 'rb') as fo: # fo = open(input_file, 'rb') 로 쓸수 있음
        data_dict = cPickle.load(fo, encoding='latin1')
    # print('a')
    return data_dict['coarse_label_names'], data_dict['fine_label_names']




def data_load(dataset_path, args):

    base_path = dataset_path
    if args.dataset == 'cifar10':
        print("| Loadding CIFAR-10 dataset\n")
        dataset_filename = base_path +'/cifar10_data.pth'

        data_cifar10    = torch.load(dataset_filename)
        train_data      = data_cifar10['train_data']
        train_labels    = data_cifar10['train_labels']

        test_data       = data_cifar10['test_data']
        test_labels     = data_cifar10['test_labels']
        num_classes     = data_cifar10['num_classes']

    elif args.dataset == 'cifar100':
        print("| Loadding CIFAR-100 dataset\n")
        dataset_filename = base_path + '/cifar100_data.pth'

        data_cifar100   = torch.load(dataset_filename)
        train_data      = data_cifar100['train_data']
        train_labels    = data_cifar100['train_fine_labels']

        test_data       = data_cifar100['test_data']
        test_labels     = data_cifar100['test_fine_labels']
        num_classes     = data_cifar100['num_classes']

    return train_data, train_labels, test_data, test_labels, num_classes






## ======================================================##
## ===================== functions ======================##
## ======================================================##

def print_Net_Setting(args):
    print('Data_set\t: {DB}'.format(DB=args.dataset))
    print('Epoch\t\t: {E}'.format(E=args.num_epochs))
    print('Lr_Drop \t: {LD}'.format(LD=args.lr_drop_epoch))
    print('Net_type\t: {NT}'.format(NT=args.net_type))
    print('Depth\t\t: {D}'.format(D=args.depth))

    if hasattr(args, 'widen_factor'):
        print('WidenFactor\t: {WF}'.format(WF=args.widen_factor))
    if hasattr(args, 'alpha'):
        print('Alpha\t\t: {a}'.format(a=args.alpha))
    if hasattr(args, 'pl'):
        print('Pl\t\t\t: {P}'.format(P=args.pl))
    if hasattr(args, 'bottleneck'):
        print('Bottle Neck\t: {BN}'.format(BN=args.bottleneck))
    if hasattr(args, 'lr'):
        print('Initial lr\t: {lr}'.format(lr=args.lr))
    if hasattr(args, 'batch_size'):
        print('Batch size\t: {BS}'.format(BS=args.batch_size))
    if hasattr(args, 'cutout'):
        print('Cutout\t\t: {C}'.format(C=args.cutout))
    if hasattr(args, 'nesterov'):
        print('nesterov\t: {N}'.format(N=args.nesterov))



def make_Dir(base_path, path_list):

    # file_name = './'
    file_name = base_path
    for i, list in enumerate(path_list):
        file_name = file_name + list
        if not os.path.isdir(file_name):
            os.mkdir(file_name)
        file_name = file_name + os.sep
    return file_name


def topk_accurcy(outputs, targets, topk_range=(1,)):
    maxk = max(topk_range)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    # _,pred = torch.topk(outputs, maxk, dim=1, largest=True, sorted=True) # 위와 같은 코드임
    pred_t = pred.t()

    target_flat = targets.view(1, -1).expand(pred_t.size())
    # self.expand_as(other) is equivalent to self.expand(other.size())
    correct = pred_t.eq(target_flat)

    topk_acc = []
    for k in topk_range:
        correct_k = correct[:k].view(-1).float().sum()
        correct_k = correct_k.data.item()
        topk_acc.append(correct_k / batch_size * 100.)

    return topk_acc




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def meanstd(data):
    ch = data.shape[3]

    mean = []
    std = []
    for i in range(ch):
        data_ch = data[:,:,:,i]
        mean.append(np.mean(data_ch))
        std.append(np.std(data_ch))
    return np.array(mean), np.array(std)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def imshow(img):
    npimg = img     # unnormalize
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0))) #img : channel, height, width --> height, width, channel 순서 변경
    plt.imshow(npimg)
    plt.show()


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img






## ======================================================##
## ============ function for calculate loss =============##
## ======================================================##

def Disjoint_loss2(predict_result, class_list_in_group, device):
    # applied grouping info

    disjoint_loss = (torch.tensor(0, dtype=torch.float32)).to(device)
    outputs=F.softmax(predict_result, dim=1)
    # print(outputs.size())
    for l_i, class_list in enumerate(class_list_in_group):
        disjoint_list_temp = []
        for i, class_idx1 in enumerate(class_list):
            for j, class_idx2 in enumerate(class_list):
                if i < j:
                    disjoint_list_temp.append(outputs[:, class_idx1] * outputs[:, class_idx2])
        disjoint_list = torch.stack(tuple(disjoint_list_temp)).t()
        disjoint_list_sum = disjoint_list.sum(dim=1)
        disjoint_loss += torch.mean(disjoint_list_sum)

    return disjoint_loss



def Disjoint_loss(predict_result):
    outputs = F.softmax(predict_result, dim=1)
    disjoint_list = torch.stack(tuple([(outputs[:, i] * outputs[:, j])
                                       for i in range(outputs.size()[1])
                                       for j in range(outputs.size()[1])
                                       if i < j])).t()
    disjoint_list_sum = disjoint_list.sum(dim=1)
    disjoint_loss = torch.mean(disjoint_list_sum)
    return disjoint_loss


# def Disjoint_loss(predict_result):
#     function_prob = nn.Softmax()
#     outputs = function_prob(predict_result)
#     disjoint_list_temp = []
#     for i in range(outputs.size()[1]):
#         # outputs.size()[1] == num_classes
#         for j in range(outputs.size()[1]):
#             if i < j:
#                 disjoint_list_temp.append(outputs[:, i] * outputs[:, j])
#     disjoint_list = torch.stack(tuple(disjoint_list_temp)).t()  # 리스트 쌓은 후 전치
#
#     disjoint_list_sum = disjoint_list.sum(dim=1)
#     # disjoint_list_sum = torch.sum(disjoint_list, dim=1)
#     disjoint_loss = torch.mean(disjoint_list_sum)
#     return disjoint_loss


def param(num_classes, num_groups):

    param = torch.Tensor(num_classes, num_groups).normal_(mean=0, std=0.01)
    # GPU 에 올릴지 말지 수정 필요
    return nn.Parameter(param, requires_grad=True)


def class_score_to_list(net_outputs, target_label):
    data_class_score = []
    data_label = []

    class_score = F.softmax(net_outputs, dim=1)
    data_class_score2 = class_score.cpu()
    data_label2 = target_label.cpu()
    for i, score in enumerate(class_score):
        data_class_score.append(score.cpu())

    for i, label in enumerate(target_label):
        data_label.append(label.cpu())

    # data_label_list = []
    # for i in range(len(data_label)):
    #     temp_label = data_label[i]
    #     data_label_list.append(temp_label.data.item())
    # data_label = data_label_list

    data_class_score = torch.stack(tuple(data_class_score))

    return data_class_score, data_label





## Grouping
def label_grouping2(net_outputs, target_label, num_classes, group_number_list, lamda_list,
                    iteration, pa_g, lr=0.1):
    ## label_grouping_with_p and return classlist with p
    # score : class score applied softmax function
    # label : data label
    # number_group : target grouping number, hierachical model
    # lamda_list : parameter for class grouping
    # pa_g : probability of belonging to group.
    pa = torch.Tensor(torch.zeros(pa_g.size()[0], pa_g.size()[1]))
    pa.data = pa_g.clone()

    data_class_score, data_label = class_score_to_list(net_outputs, target_label)
    class_count = torch.zeros(num_classes, dtype=torch.int64)

    # num_data = len(data_class_score)
    num_data = data_class_score.size()[0]
    first_class_list = []
    prob_g = []
    for i in range(num_data):
        class_count[data_label[i]] += 1

    for i in range(num_classes):
        if class_count[i] != 0:
            first_class_list.append(i)
            prob_g.append(pa[i])

    # generate pa
    prob_g = torch.stack(tuple(prob_g))
    prob_g = nn.Parameter(prob_g)

    h_level = len(group_number_list)
    ori_label = np.arange(num_classes)
    ori_label = ori_label.tolist()

    group_label_info = {'h_level': h_level,
                        'group_number_list': group_number_list,  # target number of split in each hierarchical level
                        'lamda': lamda_list,
                        'group_label': [ori_label],  # h_level * num_classes
                        'num_group': [group_number_list[0]],  # number of group in each hierarchical level
                        'num_classes_in_group': [[len(first_class_list)]],
                        # number of class in each group at each hierarchical level
                        'class_list_in_group': [[first_class_list]],  # h_level * num_group * class_list
                        }

    lamda1 = lamda_list[0]
    lamda2 = lamda_list[1]
    lamda3 = round(math.pow(0.1, lamda_list[2]), lamda_list[2])


    for h in range(h_level-1):
        temp_group_label = torch.zeros(num_classes, dtype=torch.int64)
        temp_num_classes_in_group = []
        temp_class_list_in_group = []
        num_classes_split_in_group = []
        class_list_split_in_group = []

        for g in range(group_label_info['num_group'][h]):
            class_list = group_label_info['class_list_in_group'][h][g]
            num_class_list = len(class_list)
            assert num_class_list == group_label_info['num_classes_in_group'][h][g], 'Error: num_class'

            temp_data_set_in_g = []
            temp_label_set_in_g = []

            mean_score_data = torch.zeros(num_class_list, num_classes)
            count_each_class = torch.zeros(num_class_list)

            for n_c in range(num_class_list):
                for n_d in range(num_data):
                    if class_list[n_c] == data_label[n_d]:
                        # temp_data_set_in_g.append(data_class_score[n_d])
                        # temp_label_set_in_g.append(data_label[n_d])
                        #
                        mean_score_data[n_c] += data_class_score[n_d]
                        count_each_class[n_c] = count_each_class[n_c] + 1

            for n_c in range(num_class_list):
                if count_each_class[n_c] != 0:
                    mean_score_data[n_c] = mean_score_data[n_c] / count_each_class[n_c]

            num_group_target = group_number_list[h + 1]

            split_module = GroupAssignmentLoss_V3(num_class_list, num_group_target, prob_g)
            optimizer = optim.SGD(split_module.parameters(), lr=lr, momentum=0.9)

            for epoch in range(iteration):
                optimizer.zero_grad()
                disjoint_loss = split_module(mean_score_data)
                overlap_loss = split_module.overlap_loss()
                balance_loss = split_module.balance_loss()
                total_loss = lamda1 * disjoint_loss + lamda2 * overlap_loss + lamda3 * balance_loss

                total_loss.backward(retain_graph=True)
                optimizer.step()

                # print('\n| epoch: [%5d/%5d], h_level: %2d group_number: %3d lamda1: %2d, lamda2: %2d, lamda3: %2d\n'
                #       '|d_loss: %.6f, o_loss: %.6f, b_loss: %.6f'
                #       % (epoch, iteration, h, g, lamda1, lamda2, lamda_list[2],
                #          disjoint_loss, overlap_loss, balance_loss))

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

        temp_num_group = len(temp_num_classes_in_group)
        group_label_info['num_group'].append(temp_num_group)
        group_label_info['num_classes_in_group'].append(temp_num_classes_in_group)
        group_label_info['class_list_in_group'].append(temp_class_list_in_group)

        for n_g in range(temp_num_group):
            for n_c_g in range(len(temp_class_list_in_group[n_g])):
                group_label_idx = temp_class_list_in_group[n_g][n_c_g]
                temp_group_label[group_label_idx] = n_g

        group_label_info['group_label'].append(temp_group_label.tolist())

    # # pa update
    for i, label in enumerate(first_class_list):
        pa[label] = split_module.pa[i]

    return group_label_info['class_list_in_group'][h_level-1], group_label_info['class_list_in_group'][0], pa



def label_grouping(net_outputs, target_label, num_classes, group_number_list, lamda_list, iteration):
    # score : class score applied softmax function
    # label : data label
    # number_group : target grouping number, hierachical model
    # lamda_list : parameter for class grouping


    data_class_score, data_label = class_score_to_list(net_outputs, target_label)



    class_count = torch.zeros(num_classes, dtype=torch.int64)


    # num_data = len(data_class_score)
    num_data = data_class_score.size()[0]
    first_class_list = []

    for i in range(num_data):
        class_count[data_label[i]] += 1

    for i in range(num_classes):
        if class_count[i] != 0:
            first_class_list.append(i)

    h_level = len(group_number_list)
    ori_label = np.arange(num_classes)
    ori_label = ori_label.tolist()

    group_label_info = {'h_level': h_level,
                        'group_number_list': group_number_list,  # target number of split in each hierarchical level
                        'lamda': lamda_list,
                        'group_label': [ori_label],  # h_level * num_classes
                        'num_group': [group_number_list[0]],  # number of group in each hierarchical level
                        'num_classes_in_group': [[len(first_class_list)]],
                        # number of class in each group at each hierarchical level
                        'class_list_in_group': [[first_class_list]],  # h_level * num_group * class_list
                        }

    lamda1 = lamda_list[0]
    lamda2 = lamda_list[1]
    lamda3 = round(math.pow(0.1, lamda_list[2]), lamda_list[2])


    for h in range(h_level-1):
        temp_group_label = torch.zeros(num_classes, dtype=torch.int64)
        temp_num_classes_in_group = []
        temp_class_list_in_group = []
        num_classes_split_in_group = []
        class_list_split_in_group = []

        for g in range(group_label_info['num_group'][h]):
            class_list = group_label_info['class_list_in_group'][h][g]
            num_class_list = len(class_list)
            assert num_class_list == group_label_info['num_classes_in_group'][h][g], 'Error: num_class'

            temp_data_set_in_g = []
            temp_label_set_in_g = []

            mean_score_data = torch.zeros(num_class_list, num_classes)
            count_each_class = torch.zeros(num_class_list)

            for n_c in range(num_class_list):
                for n_d in range(num_data):
                    if class_list[n_c] == data_label[n_d]:
                        # temp_data_set_in_g.append(data_class_score[n_d])
                        # temp_label_set_in_g.append(data_label[n_d])
                        #
                        mean_score_data[n_c] += data_class_score[n_d]
                        count_each_class[n_c] = count_each_class[n_c] + 1

            for n_c in range(num_class_list):
                if count_each_class[n_c] != 0:
                    mean_score_data[n_c] = mean_score_data[n_c] / count_each_class[n_c]

            num_group_target = group_number_list[h + 1]

            split_module = GroupAssignmentLoss_V2(num_class_list, num_group_target)
            optimizer = optim.SGD(split_module.parameters(), lr=0.01, momentum=0.9)

            for epoch in range(iteration):
                optimizer.zero_grad()
                disjoint_loss = split_module(mean_score_data)
                overlap_loss = split_module.overlap_loss()
                balance_loss = split_module.balance_loss()
                total_loss = lamda1 * disjoint_loss + lamda2 * overlap_loss + lamda3 * balance_loss

                total_loss.backward(retain_graph=True)
                optimizer.step()

                # print('\n| epoch: [%5d/%5d], h_level: %2d group_number: %3d lamda1: %2d, lamda2: %2d, lamda3: %2d\n'
                #       '|d_loss: %.6f, o_loss: %.6f, b_loss: %.6f'
                #       % (epoch, iteration, h, g, lamda1, lamda2, lamda_list[2],
                #          disjoint_loss, overlap_loss, balance_loss))

            # save group list
            p = split_module.p
            p_group_grob, p_group_index = torch.max(p, 1)

            for g_idx in range(num_group_target):
                class_list_split = []
                for i in range(num_class_list):
                    if p_group_index[i] == g_idx:
                        class_list_split.append(class_list[i])
                if len(class_list_split) != 0:
                    temp_num_classes_in_group.append(len(class_list_split))
                    temp_class_list_in_group.append(class_list_split)

        temp_num_group = len(temp_num_classes_in_group)
        group_label_info['num_group'].append(temp_num_group)
        group_label_info['num_classes_in_group'].append(temp_num_classes_in_group)
        group_label_info['class_list_in_group'].append(temp_class_list_in_group)

        for n_g in range(temp_num_group):
            for n_c_g in range(len(temp_class_list_in_group[n_g])):
                group_label_idx = temp_class_list_in_group[n_g][n_c_g]
                temp_group_label[group_label_idx] = n_g

        group_label_info['group_label'].append(temp_group_label.tolist())

    return group_label_info['class_list_in_group'][h_level-1]



class GroupAssignmentLoss_V3(nn.Module):
    def __init__(self, num_classes, num_groups, pa=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        if pa is not None:
            self.pa = pa
        else:
            self.pa = param(self.num_classes, self.num_groups)

        self.p = F.softmax(self.pa, dim=1)

    def forward(self, x, p=None):
        # 곱셈을 위해서 트렌스 포즈 num_classes x socre of each classes
        # --> scroe of each classes x num_classes

        x = torch.t(x)
        # x = x.t()
        # p = F.softmax(self.pa)  # group assignment probability
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        gw_score_list_temp = []  # group_weight_score_list_temp
        for i in range(self.num_groups):
            gw_score_list_temp.append(torch.sum(torch.t(x * self.p[:, i]), dim=0))
            # gw_score_list_temp.append((x * self.p[:, i]).t().sum(dim=0))

        gw_score_list = torch.stack(tuple(gw_score_list_temp))

        p_sum = torch.sum(self.p, dim=0)
        gw_score_list = torch.div(torch.t(gw_score_list), p_sum)

        gw_score_list = torch.t(gw_score_list)

        # gw_score_list =


        disjoint_loss_list_temp = []
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if i < j:
                    disjoint_loss_list_temp.append(gw_score_list[i] * gw_score_list[j])

        disjoint_loss_list = torch.stack(disjoint_loss_list_temp)
        disjoint_loss_list_sum = torch.sum(disjoint_loss_list, dim=1)
        # disjoint_loss_list_sum = disjoint_loss_list.sum(dim=1)
        disjoint_loss = torch.mean(disjoint_loss_list_sum)

        return disjoint_loss

    def overlap_loss(self, p=None):
        # p = F.softmax(self.pa)
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        overlap_loss_list_temp = []
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if i < j:
                    overlap_loss_list_temp.append(self.p[:, i] * self.p[:, j])

        overlap_loss_list = torch.stack(tuple(overlap_loss_list_temp))
        overlap_loss_list_sum = torch.sum(overlap_loss_list, dim=1)
        # overlap_loss_list_sum = overlap_loss_list.sum(dim=1)
        overlap_loss = torch.mean(overlap_loss_list_sum)

        return overlap_loss

    def balance_loss(self, p=None):
        # p = F.softmax(self.pa)
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        p_sum = torch.sum(self.p, dim=0)
        # p_sum = p.sum(dim=0)
        balance_loss_temp = p_sum * p_sum
        balance_loss_sum = torch.sum(balance_loss_temp)
        # balance_loss_sum = balance_loss_temp.sum()
        balance_loss = balance_loss_sum

        return balance_loss


class GroupAssignmentLoss_V2(nn.Module):
    def __init__(self, num_classes, num_groups):
        super().__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        # self.pa = param(self.num_classes, self.num_groups)
        self.pa = param(self.num_classes, self.num_groups)
        self.p = F.softmax(self.pa, dim=1)

    def forward(self, x, p=None):
        # 곱셈을 위해서 트렌스 포즈 num_classes x socre of each classes
        # --> scroe of each classes x num_classes

        x = torch.t(x)
        # x = x.t()
        # p = F.softmax(self.pa)  # group assignment probability
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        gw_score_list_temp = []  # group_weight_score_list_temp
        for i in range(self.num_groups):
            gw_score_list_temp.append(torch.sum(torch.t(x * self.p[:, i]), dim=0))
            # gw_score_list_temp.append((x * self.p[:, i]).t().sum(dim=0))

        gw_score_list = torch.stack(tuple(gw_score_list_temp))

        p_sum = torch.sum(self.p, dim=0)
        gw_score_list = torch.div(torch.t(gw_score_list), p_sum)

        gw_score_list = torch.t(gw_score_list)

        # gw_score_list =


        disjoint_loss_list_temp = []
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if i < j:
                    disjoint_loss_list_temp.append(gw_score_list[i] * gw_score_list[j])

        disjoint_loss_list = torch.stack(disjoint_loss_list_temp)
        disjoint_loss_list_sum = torch.sum(disjoint_loss_list, dim=1)
        # disjoint_loss_list_sum = disjoint_loss_list.sum(dim=1)
        disjoint_loss = torch.mean(disjoint_loss_list_sum)

        return disjoint_loss

    def overlap_loss(self, p=None):
        # p = F.softmax(self.pa)
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        overlap_loss_list_temp = []
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if i < j:
                    overlap_loss_list_temp.append(self.p[:, i] * self.p[:, j])

        overlap_loss_list = torch.stack(tuple(overlap_loss_list_temp))
        overlap_loss_list_sum = torch.sum(overlap_loss_list, dim=1)
        # overlap_loss_list_sum = overlap_loss_list.sum(dim=1)
        overlap_loss = torch.mean(overlap_loss_list_sum)

        return overlap_loss

    def balance_loss(self, p=None):
        # p = F.softmax(self.pa)
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        p_sum = torch.sum(self.p, dim=0)
        # p_sum = p.sum(dim=0)
        balance_loss_temp = p_sum * p_sum
        balance_loss_sum = torch.sum(balance_loss_temp)
        # balance_loss_sum = balance_loss_temp.sum()
        balance_loss = balance_loss_sum

        return balance_loss



class GroupAssignmentLoss(nn.Module):
    def __init__(self, num_classes, num_groups):
        super().__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.pa = param(self.num_classes, self.num_groups)
        self.p = F.softmax(self.pa, dim=1)

    def forward(self, x, p=None):
        # 곱셈을 위해서 트렌스 포즈 num_classes x socre to each classes
        # --> scroe to each classes x num_classes

        x = torch.t(x)
        # x = x.t()
        # p = F.softmax(self.pa)  # group assignment probability
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        gw_score_list_temp = []  # group_weight_score_list_temp
        for i in range(self.num_groups):
            gw_score_list_temp.append(torch.sum(torch.t(x * self.p[:, i]), dim=0))
            # gw_score_list_temp.append((x * self.p[:, i]).t().sum(dim=0))

        gw_score_list = torch.stack(tuple(gw_score_list_temp))


        disjoint_loss_list_temp = []
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if i < j:
                    disjoint_loss_list_temp.append(gw_score_list[i] * gw_score_list[j])

        disjoint_loss_list = torch.stack(disjoint_loss_list_temp)
        disjoint_loss_list_sum = torch.sum(disjoint_loss_list, dim=1)
        # disjoint_loss_list_sum = disjoint_loss_list.sum(dim=1)
        disjoint_loss = torch.mean(disjoint_loss_list_sum)

        return disjoint_loss

    def overlap_loss(self, p=None):
        # p = F.softmax(self.pa)
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        overlap_loss_list_temp = []
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if i < j:
                    overlap_loss_list_temp.append(self.p[:, i] * self.p[:, j])

        overlap_loss_list = torch.stack(tuple(overlap_loss_list_temp))
        overlap_loss_list_sum = torch.sum(overlap_loss_list, dim=1)
        # overlap_loss_list_sum = overlap_loss_list.sum(dim=1)
        overlap_loss = torch.mean(overlap_loss_list_sum)

        return overlap_loss

    def balance_loss(self, p=None):
        # p = F.softmax(self.pa)
        if p is not None:
            self.p = p
        else:
            self.p = F.softmax(self.pa, dim=1)  # group assignment probability

        p_sum = torch.sum(self.p, dim=0)
        # p_sum = p.sum(dim=0)
        balance_loss_temp = p_sum * p_sum
        balance_loss_sum = torch.sum(balance_loss_temp)
        # balance_loss_sum = balance_loss_temp.sum()
        balance_loss = balance_loss_sum

        return balance_loss


## ======================================================##
## ========= function for network initialization ========##
## ======================================================##
# def get_filename(args, num_classes, group_label_info):
#     h_level = args.h_level
#
#     # num_group = group_label_info['num_group']
#     # number_split_target = group_label_info['number_split_target']
#     # lamda = group_label_info['lamda']
#     num_group = num_classes
#     lamda_value = group_label_info['lamda']
#     str_depth = '-D'
#     str_num_group = '-G'
#     str_lamda_value = '-L'
#     str_cutout = '-Cutout'
#
#     for d in range(len(args.depth)):
#         str_depth = str_depth + '_{D:02d}'.format(D=args.depth[d])
#     for n_g in range(len(num_group)):
#         str_num_group = str_num_group + '_{G:02d}'.format(G=num_group[n_g])
#
#     str_lamda3 = '_10e-{l3}'.format(l3=lamda_value[2])
#     for l_v in range(len(lamda_value)-1):
#         str_lamda_value = str_lamda_value + '_{L:02d}'.format(L=lamda_value[l_v])
#     str_lamda_value = str_lamda_value + str_lamda3
#
#
#
#     if args.net_type == 'RefineNet_WRN':
#         file_name = args.net_type + '-' +'H{H}'.format(H=h_level) \
#                     + str_depth + 'x' + str(args.widen_factor) + str_num_group + str_lamda_value
#
#     elif args.net_type == 'RefineNet_RN' or args.net_type == 'RefineNet_PRN':
#         file_name = args.net_type + '-' +'H{H}'.format(H=h_level) \
#                                         + str_depth + str_num_group + str_lamda_value
#
#         if args.bottleneck == True:
#             file_name = file_name + '_bottleneck'
#
#     elif args.net_type == 'RefineNet_PYN':
#         file_name = args.net_type + '-' +'H{H}'.format(H=h_level) \
#                     + str_depth + '_alpha' + str(args.alpha) + str_num_group + str_lamda_value
#
#         if args.bottleneck == True:
#             file_name = file_name + '_bottleneck'
#
#     else:
#         print('Error : Network should be either [RefineNet_WRN / RefineNet_RN / RefineNet_PRN/RefineNet_PYN')
#         print('Func.: get_filename')
#         sys.exit(0)
#
#     if hasattr(args, 'cutout'):
#         if args.cutout == True:
#             file_name = file_name + str_cutout
#
#     return file_name

def error_message_network_name():
    print('Error : Network should be either [RefineNet_WRN / RefineNet_RN / RefineNet_PRN / RefineNet_PYN(_SD)/ ] or ')
    print('[wide_resnet / resnet / preact_resnet / pyramid_resnet(_SD)]')


def get_filename_for_split(args, group_label_info, num_classes):

    h_level = group_label_info['h_level']
    num_group = group_label_info['num_group']
    number_group_target_list = group_label_info['number_group_target_list']
    lamda = group_label_info['lamda']

    # generate string about split info.
    str_num_group = '_G'
    for n_g_l in range(len(num_group)):
        str_num_group = str_num_group + '-{G:02d}'.format(G=num_group[n_g_l])

    str_split_target = '_S'
    for s_t in range(len(number_group_target_list)):
        str_split_target = str_split_target + '-{S:02d}'.format(S=number_group_target_list[s_t])

    str_lamda_value = '_L'
    lamda3_str = '-(10e-{l3})'.format(l3=lamda[2])
    for l_v in range(len(lamda) - 1):
        str_lamda_value = str_lamda_value + '-{L:02d}'.format(L=lamda[l_v])
    str_lamda_value = str_lamda_value + lamda3_str

    str_split_info = str_num_group + str_split_target + str_lamda_value

    # generate file name
    # if args.net_type == 'wide_resnet':
    #     file_name = get_filename(args) + '_h{H}'.format(H=h_level) + str_split_info
    #
    # elif args.net_type == 'resnet':
    #     file_name = get_filename(args) + '_h{H}'.format(H=h_level) + str_split_info
    #
    # elif args.net_type == 'preact_resnet':
    #     file_name = get_filename(args) + '_h{H}'.format(H=h_level) + str_split_info

    file_name = get_filename(args, num_classes) + '_h{H}'.format(H=h_level) + str_split_info

    return file_name



def get_filename(args, num_classes, group_label_info=None):
    h_level = args.h_level

    num_group = num_classes
    str_depth = '-D'
    str_num_group = '-G'
    str_lamda_value = '-L'
    str_cutout = '-Cutout'

    if group_label_info is not None:
        lamda_value = group_label_info['lamda']
        str_lamda3 = '_10e-{l3}'.format(l3=lamda_value[2])
        for l_v in range(len(lamda_value) - 1):
            str_lamda_value = str_lamda_value + '_{L:02d}'.format(L=lamda_value[l_v])
        str_lamda_value = str_lamda_value + str_lamda3

    for d in range(len(args.depth)):
        str_depth = str_depth + '_{D:02d}'.format(D=args.depth[d])
    for n_g in range(len(num_group)):
        str_num_group = str_num_group + '_{G:02d}'.format(G=num_group[n_g])



    if args.net_type.startswith('RefineNet'):
        file_name = args.net_type + '-' +'H{H}'.format(H=h_level) \
                    + str_depth
        if hasattr(args, 'widen_factor'):
            file_name = file_name + 'x' + str(args.widen_factor)
        if hasattr(args, 'alpha'):
            file_name = file_name + '_alpha' + str(args.alpha)
        if hasattr(args, 'pl'):
            file_name = file_name + '_pl' + str(args.pl)

        file_name = file_name + str_num_group + str_lamda_value

        if hasattr(args, 'bottleneck'):
            if args.bottleneck == True:
                file_name = file_name + '_bottleneck'

    else:
        file_name = args.net_type + '-' + 'H{H}'.format(H=h_level) \
                    + str_depth
        if hasattr(args, 'widen_factor'):
            file_name = file_name + 'x' + str(args.widen_factor)
        if hasattr(args, 'alpha'):
            file_name = file_name + '_alpha' + str(args.alpha)
        if hasattr(args, 'pl'):
            file_name = file_name + '_pl' + str(args.pl)

        file_name = file_name + str_num_group

        if hasattr(args, 'bottleneck'):
            if args.bottleneck == True:
                file_name = file_name + '_bottleneck'

    if hasattr(args, 'cutout'):
        if args.cutout == True:
            file_name = file_name + str_cutout

    return file_name


def get_network_init_weight(args, num_classes):
    # if args.h_level == 2:

    ## Refine Network
    if args.net_type == 'RefineNet_WRN':
        net = RWRN.Wide_ResNet(args.depth, num_classes, args.dataset, args.widen_factor, args.dropout)
        net.apply(RWRN.weight_init)

    elif args.net_type == 'RefineNet_RN':
        net = RRN.RefineNet_ResNet(args.depth, num_classes, args.dataset, bottleneck=args.bottleneck)
        net.apply(RRN.weight_init)

    # elif args.net_type == 'preact_resnet':
    #     net = RPRNT.PreResNet(args.depth, num_classes)

    elif args.net_type == 'RefineNet_PRN':
        net = RPRN.RefineNet_PreActResNet(args.depth, num_classes, args.dataset, bottleneck=args.bottleneck)
        net.apply(RPRN.weight_init)

    elif args.net_type == 'RefineNet_PYN':
        net = RPYN.RefineNet_PyramidNet(args.depth, num_classes, args.dataset, args.alpha, bottleneck=args.bottleneck)
        net.apply(RPYN.weight_init)

    elif args.net_type == 'RefineNet_PYN_SD':
        net = RPYN_SD.RefineNet_PyramidNet_ShakeDrop(args.depth, num_classes, args.dataset, args.alpha,
                                                     bottleneck=args.bottleneck, pl=args.pl)
        net.apply(RPYN_SD.weight_init)


    ## Base Network

    elif args.net_type == 'wide_resnet':
        net = WRN.Wide_ResNet(args.depth, num_classes, args.widen_factor, args.dropout)
        net.apply(WRN.weight_init)

    elif args.net_type == 'resnet':
        net = RN.ResNet(args.depth, num_classes, args.dataset, bottleneck=args.bottleneck)
        net.apply(RN.weight_init)


    elif args.net_type == 'preact_resnet':
        net = PRN.PreActResNet(args.depth, num_classes, args.dataset, bottleneck=args.bottleneck)
        net.apply(PRN.weight_init)

    elif args.net_type == 'pyramid_net':
        net = PYN.PyramidNet(args.depth, num_classes, args.dataset, args.alpha, bottleneck=args.bottleneck)
        net.apply(PYN.weight_init)

    elif args.net_type == 'pyramid_net_SD':
        net = PYN_SD.PyramidNet_ShakeDrop(args.depth, num_classes, args.dataset, args.alpha,
                                          bottleneck=args.bottleneck, pl=args.pl)
        net.apply(PYN_SD.weight_init)

    else:
        error_message_network_name()
        print('Func.: get_network')
        sys.exit(0)

    return net


def learning_rate(args, epoch):

    init_lr = args.lr
    lr = init_lr
    gamma = args.gamma
    lr_drop_epoch = args.lr_drop_epoch
    for i in range(len(lr_drop_epoch)):
        if epoch >= lr_drop_epoch[i]:
            lr = init_lr * math.pow(gamma, i + 1)

    return lr

def momentum_weightdecay(args):
    ## For Refine network
    if args.net_type == 'RefineNet_WRN':
        mmt = float(0.9)
        w_decay = float(5e-4)
    elif args.net_type == 'RefineNet_RN':
        mmt = float(0.9)
        w_decay = float(1e-4)

    elif args.net_type == 'RefineNet_PRN':
        mmt = float(0.9)
        w_decay = float(1e-4)

    elif args.net_type == 'RefineNet_PYN':
        mmt = float(0.9)
        w_decay = float(1e-4)

    elif args.net_type == 'RefineNet_PYN_SD':
        mmt = float(0.9)
        w_decay = float(1e-4)


    ## For base network
    elif args.net_type == 'wide_resnet':
        mmt = float(0.9)
        w_decay = float(5e-4)
    elif args.net_type == 'resnet':
        mmt = float(0.9)
        w_decay = float(1e-4)

    elif args.net_type == 'preact_resnet':
        mmt = float(0.9)
        w_decay = float(1e-4)

    elif args.net_type == 'pyramid_net':
        mmt = float(0.9)
        w_decay = float(1e-4)

    elif args.net_type == 'pyramid_net_SD':
        mmt = float(0.9)
        w_decay = float(1e-4)

    else:
        error_message_network_name()
        print('Func.: momentum_weightdecay')
        sys.exit(0)

    return mmt, w_decay




def info_plot_acc(input_args):
    info = {
            'net_type' : input_args.net_type
    }


def save_network(input_args, state_dict, best_top1_acc, best_top5_acc, epoch, summary, group_label_info=None):
    args = copy.deepcopy(input_args)
    state = {
            'args'          : args,
            'state_dict'    : state_dict,
            'best_top1_acc' : best_top1_acc,
            'best_top5_acc' : best_top5_acc,
            'epoch'         : epoch,
            'summary'       : summary,
            'group_label_info': group_label_info
            }


    return state
# def load_param(input_args, checkpoint):
#     # Load checkpoint data
#     args = copy.deepcopy(input_args)
#     print("| Resuming from checkpoint...")
#
#     args                = checkpoint['args']
#     args.start_epoch    = checkpoint['epoch'] + 1
#     state_dict          = checkpoint['state_dict']
#     best_top1_acc       = checkpoint['best_top1_acc']
#     best_top5_acc       = checkpoint['best_top5_acc']
#     summary             = checkpoint['summary']
#     plot_acc_info       = checkpoint['plot_acc_info']
#
#
#     return args, state_dict, best_top1_acc, best_top5_acc, summary, plot_acc_info

def load_param(input_args, checkpoint):
    # Load checkpoint data
    args = copy.deepcopy(input_args)
    print("| Resuming from checkpoint...")

    args                = checkpoint['args']
    args.start_epoch    = checkpoint['epoch'] + 1
    state_dict          = checkpoint['state_dict']
    best_top1_acc       = checkpoint['best_top1_acc']
    best_top5_acc       = checkpoint['best_top5_acc']
    summary             = checkpoint['summary']

    if args.net_type.startswith('RefineNet'):
        group_label_info    = checkpoint['group_label_info']
        return args, state_dict, best_top1_acc, best_top5_acc, summary, group_label_info

    else:
        return args, state_dict, best_top1_acc, best_top5_acc, summary


def save_class_score(net, dataloader, num_dataset, device):

    data_class_score = []
    data_label = []

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            class_score = F.softmax(outputs, dim=1)

            for i, score in enumerate(class_score):
                data_class_score.append(score.cpu())

            for i, label in enumerate(targets):
                data_label.append(label.cpu())

            print('| Progress: Iter[{current_idx_set:3d} / {total_num_set:3d}]'
                  .format(current_idx_set=batch_idx + 1, total_num_set=math.ceil((num_dataset/dataloader.batch_size))))

        data_class_score = torch.stack(tuple(data_class_score))
        # print(data_class_score)

    return data_class_score, data_label


def save_class_score_without_softmax(net, dataloader, num_dataset, device):

    data_class_score = []
    data_label = []

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            for i, score in enumerate(outputs):
                data_class_score.append(score.cpu())

            for i, label in enumerate(targets):
                data_label.append(label.cpu())

            print('| Progress: Iter[{current_idx_set:3d} / {total_num_set:3d}]'
                  .format(current_idx_set=batch_idx + 1, total_num_set=math.ceil((num_dataset/dataloader.batch_size))))

        data_class_score = torch.stack(tuple(data_class_score))

    return data_class_score, data_label

def network_hyperparameter_check(args, args2, lr_drop_revise=False):
    print('Checking network hyparameter......')
    if len(args.depth) == len(args2.depth):
        assert args.depth           == args2.depth,         'Please check network depth'
    else:
        assert args.depth[0]        == args2.depth[0],  'Please check network depth'

    assert args.num_epochs          == args2.num_epochs,    'Please check training epoch(num_epoch)'
    assert args.gamma               == args2.gamma,         'please check gamma'
    # assert args.batch_size          == args2.batch_size,    'please check batch size'

    if hasattr(args, 'lr_drop_epoch') and lr_drop_revise == False:
        assert args.lr_drop_epoch   == args2.lr_drop_epoch, 'please check lr drop epoch'
    if hasattr(args, 'bottleneck'):
        assert args.bottleneck      == args2.bottleneck,    'please check bottleneck'
    if hasattr(args, 'alpha'):
        assert args.alpha           == args2.alpha,         'please check alpha value'
    if hasattr(args, 'widen_factor'):
        assert args.widen_factor    == args2.widen_factor,  'Please check widen factor'
    if hasattr(args, 'pl'):
        assert args.pl              == args2.pl,            'please check pl value'

    if args.net_type == args2.net_type:
        if hasattr(args, 'cutout'):
            assert args.cutout          == args2.cutout,    'please check applied cutout'
        assert args.lr                  == args2.lr,        'Please check learning rate: lr'

        if args.net_type == 'RefineNet_PYN' or args.net_type == 'RefineNet_PYN_SD' \
           or args.net_type == 'pyramid_net' or args.net_type == 'pyramid_net_SD':  # BaseNet: Pyramid net

            assert args.nesterov == args2.nesterov, 'Please check nesterov that option for SGD'
            if hasattr(args, 'nesterov') and args.net_type == args2.net_type:
                assert args.nesterov        == args2.nesterov,  'Please check nesterov that option for SGD'

