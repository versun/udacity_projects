import argparse
import os
parser = argparse.ArgumentParser(description='用数据集训练新的网络\n用法：python train.py data_dir')
parser.add_argument('data_dir', 
                    metavar='Data_dir', 
                    type=str, 
                    help='数据集的路径')
parser.add_argument('--save_dir',
                    metavar='    ',
                    type=str,
                    help="保存检查点的路径(default='ckp.pth')",
                    dest='save_dir',
                    default='ckp.pth')
parser.add_argument('--arch',
                    choices=['vgg13','vgg16'],
                    type=str,
                    metavar='    ',
                    help="架构(default='vgg16')",
                    dest='arch',
                    default='vgg16')
parser.add_argument('--learning_rate',
                    metavar='    ',
                    type=float,
                    help='学习速率(default=0.001)',
                    dest='learning_rate',
                    default=0.001)
parser.add_argument('--hidden_units',
                    metavar='    ',
                    type=int,
                    help='隐藏层数量(default=10240)',
                    dest='hidden_units',
                    default=10240)
parser.add_argument('--epochs',
                    metavar='    ',
                    type=int,
                    help='学习次数(default=3)',
                    dest='epochs',
                    default=3)
parser.add_argument('--gpu',
                    help='使用GPU(default=False)',
                    metavar='False',
                    dest='gpu',
                    default=False,
                    type=bool,
                    nargs='?',
                    const=True
                    )

args = parser.parse_args()

if os.path.exists(args.data_dir):
    
    print("data_directory:\t",args.data_dir)
    print("save_directory:\t",args.save_dir)
    print("arch:\t\t",args.arch)
    print("learning_rate:\t",args.learning_rate)
    print("hidden_units:\t",args.hidden_units)
    print("epochs:\t\t",args.epochs)
    print("gpu:\t\t",args.gpu)

    import main
    main.do_train(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
else:
    print("File is doesn't exist")

