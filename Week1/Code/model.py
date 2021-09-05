# nn.ModuleList()
# debug时重点看shape和_modules

# 注意kernal_size=3对应padding=1，而stride仅仅改变的是下采样的尺度

import torch
import torch.nn as nn
from torch.nn.modules import padding

class DBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, bn_act=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = 1 if kernel_size==3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x))) if self.use_bn_act else self.conv(x)
    
class Res_unit(nn.Module): # Res_unit不改变通道数
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            DBL(channels, channels//2,1),
            DBL(channels//2, channels,3),
        )
    
    def forward(self, x):
        return self.layers(x)+x
        
class Res_Blocks(nn.Module): # Res_Blocks通过刚进入时的DBL块同时改变size和channel
    def __init__(self, in_channels, out_channels, nums, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nums = nums
        self.layers = nn.ModuleList()
        self.layers += [DBL(in_channels,out_channels,3,stride=2)]
        for _ in range(nums):
            self.layers+=[Res_unit(out_channels)]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Prediction(nn.Module):
    '''
    def __init__(self, channels, num_classes, **kwargs):
        super().__init__()
        self.layer
    '''
    pass
    
    
class backbone(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()   
        self.layers.append(DBL(in_channels,32,3))
        in_channels = 32
        num_list = [1,2,8,8,4]
        for num in num_list:
            self.layers.append(Res_Blocks(in_channels, 2*in_channels, num))
            in_channels = 2*in_channels
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        return x
        
        
class Yolov3(nn.Module):
    '''
    def __init__(self, channels, num_classes, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()    
    '''
    pass


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 256
    model = backbone(num_classes=num_classes)
    x = torch.randn((2,3,IMAGE_SIZE,IMAGE_SIZE)) #(N,C,H,W)
    out = model(x)
    print("out:", out.shape)
    print("success!")
    