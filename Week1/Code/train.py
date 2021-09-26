import torch
from data import transforms
from models.model import Yolov3
from data.dataset import VOC2012DataSet
import argparse
import os

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]), #依概率p垂直翻转
        "val": transforms.Compose([transforms.ToTensor()])
    }
    
    VOC_root = args.data_path
    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], "train.txt")
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=num_workers,
                                                    collate_fn=train_data_set.collate_fn) # 不用默认方法，而是打包
    
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=num_workers,
                                                      collate_fn=train_data_set.collate_fn)
    
    model = Yolov3(num_classes = args.num_classes)
    model.to(device)
    
    #x = torch.randn((2,3,IMAGE_SIZE,IMAGE_SIZE)) #(N,C,H,W)
    for i, data in enumerate(train_data_loader,0): # i是序列脚标，data是具体数据
        inputs, labels = data # 这边inputs的尺寸不一样，而且inputs是一个数组，inputs需要进行预处理
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
    print("out:", outputs.shape)
    print("success!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser( description=__doc__ )
    # 命令行选项
    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data_path', default='./', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output_dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    main(args)
    print(args)
    print("hhh")
