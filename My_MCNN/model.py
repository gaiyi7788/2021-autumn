import torch
import torch.nn as nn
import os
from dataset import CrowdDataset

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, bn_act=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if kernel_size==3:
            padding = 1 
        elif kernel_size==5:
            padding = 2
        elif kernel_size==7:
            padding = 3
        elif kernel_size==9:
            padding = 4
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = not bn_act)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x))) if self.use_bn_act else self.conv(x)


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        #不同size的卷积记得改变padding值，才能保证尺寸不变
        self.branch1 = nn.Sequential( 
            Conv(3,16,9),
            self.maxpool,
            Conv(16,32,7),
            self.maxpool,
            Conv(32,16,7),
            Conv(16,8,7)
        )
        self.branch2 = nn.Sequential(
            Conv(3,20,7),
            self.maxpool,
            Conv(20,40,5),
            self.maxpool,
            Conv(40,20,5),
            Conv(20,10,5)
        )
        self.branch3 = nn.Sequential(
            Conv(3,24,5),
            self.maxpool,
            Conv(24,48,3),
            self.maxpool,
            Conv(48,24,3),
            Conv(24,12,3)
        )
        self.fusion = nn.Conv2d(30,1,1)
        
    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat((x1,x2,x3),1)
        x = self.fusion(x)
        return x
        
        
if __name__=="__main__":
    dataset_root = "dataset/ShanghaiTech/part_A/train_data"
    img_root = os.path.join(dataset_root,'images')
    gt_dmap_root = os.path.join(dataset_root,'gt_dmaps')
    dataset = CrowdDataset(img_root,gt_dmap_root,4)
    model = MCNN()
    for i,(img,gt_dmap) in enumerate(dataset):
        output = model(img)