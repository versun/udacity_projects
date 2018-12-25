import argparse
import os
parser = argparse.ArgumentParser(description='预测图像类别\n 基本用法：python predict.py image_path checkpoint_path')
parser.add_argument('image_path', 
                    metavar='image path', 
                    type=str, 
                    default='',
                    help='需要预测的图像路径')
parser.add_argument('checkpoint_path', 
                    metavar='checkpoint path', 
                    type=str, 
                    default='ckp.pth',
                    help="导入检查点并加载预测模型(default='ckp.pth')")
parser.add_argument('--top_k',
                    metavar='    ',
                    type=int,
                    help='设置top_k类别数量(default=5)',
                    dest='top_k',
                    default=5)
parser.add_argument('--category_names',
                    type=str,
                    metavar='    ',
                    help="类别到真实名称的映射文件(default='cat_to_name.json')",
                    dest='category_names',
                    default='cat_to_name.json')
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


if os.path.isfile(args.image_path) and os.path.isfile(args.checkpoint_path):
    
    print("image_path:\t",args.image_path)
    print("checkpoint_path:",args.checkpoint_path)
    print("top_k:\t\t",args.top_k)
    print("category_names:\t",args.category_names)
    print("GPU:\t\t",args.gpu)

    import main
    main.do_predict(args.image_path,args.checkpoint_path,args.top_k,args.category_names,args.gpu)
else:
    print("File is doesn't exist")
