# nn.ModuleList()
# debug时重点看shape和_modules

# 注意kernal_size=3对应padding=1，而stride仅仅改变的是下采样的尺度

# 记录一下python的 


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

class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pred = nn.Sequential(
            DBL(in_channels,out_channels,1),
            DBL(out_channels,2*out_channels,3),
            DBL(2*out_channels,out_channels,1),
            DBL(out_channels,2*out_channels,3),
            DBL(2*out_channels,out_channels,1)
        )
            
    def forward(self, x):
        x = self.pred(x)
        return x

class Prediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            DBL(in_channels,2*in_channels,3),
            DBL(2*in_channels,3*(num_classes+5),1,bn_act=False)
        )
    
    def forward(self, x):
        x = self.pred(x)
        return x
    
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
        return x
        
        
class Yolov3(nn.Module):
    def __init__(self, num_classes=20, channels=3, **kwargs):
        super().__init__()
        self.backbone = backbone(num_classes=num_classes)
        self.feature_maps = []   
        self.outputs = [] 
        self.layers = nn.ModuleList()
        self.layers += [
            YoloBlock(1024, 512), Prediction(512, num_classes), DBL(512,256,1),
            YoloBlock(768, 256), Prediction(256, num_classes), DBL(256,128,1),
            YoloBlock(384, 128), Prediction(128, num_classes)
        ]
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        backbone_modules = self.backbone.layers._modules
        for name in backbone_modules:
            x = backbone_modules[name](x)
            if name == '3' or name == '4':
                self.feature_maps.append(x)
        layers_names = ['0','1','2','3','4','5','6','7']
        for name in range(8):
            print(name)
            if name == 0 or name == 3 or name == 6:
                x = self.layers[name](x)
            elif name == 1 or name == 4 or name == 7:
                self.outputs.append(self.layers[name](x)) 
            elif name == 2 or name == 5:
                x = self.upsample(self.layers[name](x))
                # dim=1,[N,C,H,W]
                x = torch.cat([x,self.feature_maps[-1]],dim = 1)
                self.feature_maps.pop()
                        
        print("hhh")
        
        return x


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 256
    model = Yolov3(num_classes=num_classes)
    x = torch.randn((2,3,IMAGE_SIZE,IMAGE_SIZE)) #(N,C,H,W)
    out = model(x)
    print("out:", out.shape)
    print("success!")
    