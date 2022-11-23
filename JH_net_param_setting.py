import argparse
import sys

def net_param_setting(net_type, dataset):
    # =====================================================================================================================#
    # ==================================== Hyper Parameter of Network =====================================================#
    # =====================================================================================================================#

    depth = [200, 50]
    # dataset = 'imagenet'

    if net_type.startswith('RefineNet') == False:
        depth = [depth[0]]

    if net_type == 'pyramid_net' or net_type == 'pyramid_net_SD':  # BaseNet: Pyramid net
        nesterov = True
    else:
        nesterov = False
    # depth for RefineNet_WRN : 16, 22, 28, 40
    # 28 x 12, 28x10

    # depth for resnet, preact resnet, pyramid net
    # depth 20, 32, 44, 56, 110, 164, 1202: BasicBlock
    # 110, 164

    # net block type in the paper for data set 'image net'=================================================================#
    # depth 18, 34       : BasicBlock
    # depth 50, 101, 152 : Bottleneck

    # pyramid net
    # alpha in the paper
    # depth-alpha : 110-48, 110-84, 110-270
    # depth-alpha (bottleneck) : 164-270, 200-240, 236-220, 272-200

    if net_type == 'RefineNet_WRN' or net_type == 'wide_resnet':    # BaseNet: Wide resnet

        parser = argparse.ArgumentParser(description='Network Parameter Setting')

        parser.add_argument('--net_type',           default=net_type,       type=str,   help='model')
        parser.add_argument('--depth',              default=depth,          type=int,   help='depth of model')

        parser.add_argument('--num_epochs',         default=200,            type=int,   help='total number of train epoch')

        if dataset == 'cifar10' or dataset == 'cifar100':
            lr = 0.1
            lr_drop_epoch = [60, 120, 160]
        if dataset == 'imagenet':
            assert dataset == 'imagenet', 'please check paper'

        parser.add_argument('--lr',                 default=lr,             type=float, help='learning_rate')
        parser.add_argument('--gamma',              default=0.2,            type=float, help='weight for learning_rate on lr drop epoch')
        parser.add_argument('--widen_factor',       default=10,             type=int,   help='width of model')
        parser.add_argument('--dropout',            default=0.3,            type=float, help='dropout_rate')
        parser.add_argument('--lr_drop_epoch',      default=lr_drop_epoch,  type=int,   help='step for learning rate down')



    elif net_type == 'RefineNet_RN' or net_type == 'resnet':   # BaseNet: Resnet

        parser = argparse.ArgumentParser(description='Network Parameter Setting')

        parser.add_argument('--net_type',           default=net_type,       type=str,   help='model')
        parser.add_argument('--depth',              default=depth,          type=int,   help='depth of model')
        parser.add_argument('--num_epochs',         default=450,            type=int,   help='total number of train epoch')

        if dataset == 'cifar10' or dataset == 'cifar100':
            lr = 0.1
            lr_drop_epoch = [150, 250, 350]
        if dataset == 'imagenet':
            assert dataset == 'imagenet', 'please check paper'

        parser.add_argument('--lr',                 default=lr,             type=float, help='learning_rate')
        parser.add_argument('--gamma',              default=0.1,            type=float, help='weight for learning_rate on lr drop epoch')
        parser.add_argument('--lr_drop_epoch',      default=lr_drop_epoch,  type=int,   help='step for learning rate down')
        parser.add_argument('--bottleneck',         default=True,                       help='choose bottleneck option')



    elif net_type == 'RefineNet_PRN' or net_type == 'preact_resnet':  # BaseNet: Preact resnet

        parser = argparse.ArgumentParser(description='Network Parameter Setting')

        parser.add_argument('--net_type',           default=net_type,       type=str,   help='model')
        parser.add_argument('--depth',              default=depth,          type=int,   help='depth of model')


        if dataset == 'cifar10' or dataset == 'cifar100':
            lr = 0.1
            lr_drop_epoch = [2, 250, 350]
            # lr_drop_epoch = [150, 250, 350]
        if dataset == 'imagenet':
            lr = 0.1
            lr_drop_epoch = [30, 60]
            # assert dataset == 'imagenet', 'please check paper'
        if dataset.startswith('cifar'):
            num_epochs = 450
        elif dataset == 'imagenet':
            num_epochs = 120
        parser.add_argument('--num_epochs',         default=num_epochs, type=int, help='total number of train epoch')
        parser.add_argument('--lr',                 default=lr,             type=float, help='learning_rate')
        parser.add_argument('--gamma',              default=0.1,            type=float, help='weight for learning_rate on lr drop epoch')
        parser.add_argument('--lr_drop_epoch',      default=lr_drop_epoch,  type=int,   help='step for learning rate down')
        parser.add_argument('--bottleneck',         default=True,                      help='choose bottleneck option')



    elif net_type == 'RefineNet_PYN' or net_type == 'RefineNet_PYN_SD' \
            or net_type == 'pyramid_net' or net_type == 'pyramid_net_SD':  # BaseNet: Pyramid net

        parser = argparse.ArgumentParser(description='Network Parameter Setting')

        parser.add_argument('--net_type',           default=net_type,       type=str,   help='model')
        parser.add_argument('--depth',              default=depth,          type=int,   help='depth of model')



        if net_type == 'RefineNet_PYN' or net_type == 'pyramid_net':
            if dataset == 'cifar10':
                lr = 0.25
                lr_drop_epoch = [150, 225]
            elif dataset == 'cifar100':
                lr = 0.25
                lr_drop_epoch = [150, 225]
            elif dataset == 'imagenet':
                lr = 0.1
                # lr_drop_epoch = [60, 90, 105]
                lr_drop_epoch = [30, 60, 90]
        else:
            if dataset == 'cifar10':
                lr = 0.25
                lr_drop_epoch = [150, 225]
            elif dataset == 'cifar100':
                lr = 0.25
                lr_drop_epoch = [150, 225]
            elif dataset == 'imagenet':
                lr = 0.05
                # lr_drop_epoch = [60, 90, 105]
                lr_drop_epoch = [30, 60, 90]
        if dataset.startswith('cifar'):
            num_epochs = 300
        elif dataset == 'imagenet':
            num_epochs = 120

        parser.add_argument('--num_epochs', default=num_epochs, type=int, help='total number of train epoch')
        parser.add_argument('--lr',                 default=lr,             type=float, help='learning_rate')
        # cifar 10 : lr = 0.1, cifar 100: lr = 0.5
        parser.add_argument('--gamma',              default=0.1,            type=float, help='weight for learning_rate on lr drop epoch')
        parser.add_argument('--lr_drop_epoch',      default=lr_drop_epoch,  type=int,   help='step for learning rate down')
        parser.add_argument('--alpha',              default=300,            type=int,   help='addition-based widening step factor')
        parser.add_argument('--bottleneck',         default=True,                       help='choose bottleneck option')
        if net_type == 'pyramid_net_SD' or net_type == 'RefineNet_PYN_SD':
            # if depth[0] > 50:
            #     pl = 0.35
            # else:
            #     pl = 0.05
            if dataset.startswith('cifar'):
                pl = 0.25
            elif dataset == 'imagenet':
                pl = 0.9
            parser.add_argument('--pl',         default=pl,             type=float, help='initial parameter for Bernoulli random variable.')




    else:
        print('Error : Network should be either [RefineNet_WRN, RefineNet_RN, RefineNet_PRN, RefineNet_PYN')
        print('Func.: Parser args')
        sys.exit(0)

    # =====================================================================================================================#
    # settings for common parameter ======================================================================================#
    parser.add_argument('--h_level',                default=len(depth),     type=int,   help='herarch')
    parser.add_argument('--start_epoch',            default=1,              type=int,   help='start_epoch')
    parser.add_argument('--batch_size',             default=36,            type=int,   help='batch size')
    parser.add_argument('--test_batch_size',        default=170,            type=int,   help='test_batch_size')

    # apply cutout ========================================================================================================#
    if dataset.startswith('cifar'):
        cutout_length = 16
    elif dataset == 'imagenet':
        cutout_length = 112

    parser.add_argument('--cutout',                 default=False,                       help='apply cutout')
    parser.add_argument('--n_holes',                default=1,              type=int,   help='number of holes to cut out from image')
    parser.add_argument('--cutout_length',          default=cutout_length,             type=int,   help='length of the holes')
    parser.add_argument('--nesterov',               default=nesterov,                   help='option for SGD')

    parser.add_argument('--dataset',                default=dataset,        type=str,   help='dataset = [cifar10/cifar100]')
    parser.add_argument('--topk',                   default=5,              type=int,   help='')
    parser.add_argument('--resume',                 default=False,                      help='resume from checkpoint')
    parser.add_argument('--disjoint',               default=False,                      help='Apply disjoint between classes')
    parser.add_argument('--testOnly',               default=False,                      help='Test mode with the saved model')
    parser.add_argument('--save_class_score',       default=False,                      help='Save class score with the saved model')

    return parser.parse_args()