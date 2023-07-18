'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser() #类似初始化一个parser方便后续操作，括号内可以输入描述：argparse.ArgumentParser(description='Description of your program')

    parser.add_argument(
        '--data_root',
        default='./data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--img_list',
        default='./data/train.txt',
        type=str,
        help='Path for image list file')
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
    default=56,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=448,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=448,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.'
    )
    parser.add_argument(
        '--pretrain_path',
        default='pretrain/resnet_50.pth',
        type=str,
        help=
        'Path for pretrained model.'
    )
    parser.add_argument(
        '--new_layer_names',
        #default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        default=['conv_seg'],
        type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    #action='store_true': 这个参数指定了当命令行中出现了--ci_test这个标志时，该参数的值应该设置为True。如果不使用action='store_true'，则该参数将期望接收一个值（例如：--ci_test value），而不是一个简单的开关。
    args = parser.parse_args()
    args.save_folder = "./trails/models/{}_{}".format(args.model, args.model_depth)
    #"./trails/models/{}_{}".format(args.model, args.model_depth): 这是字符串格式化操作。在这里，我们使用了一个字符串模板，其中用花括号 {} 表示一个或多个待替换的占位符。format() 方法用于将占位符替换为实际的值。
    
    return args
