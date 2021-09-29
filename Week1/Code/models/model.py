# nn.ModuleList()
# debug时重点看shape和_modules

# 注意kernal_size=3对应padding=1，而stride仅仅改变的是下采样的尺度

# 记录一下python的 


import torch
import torch.nn as nn
from data.transform import Transform
#from torch.nn.modules import padding
import numpy as np
import cv2

def tensor2img(tensor):  #将tensor可视化，但好像不太好用
    array1=tensor[0].cpu().numpy()#将tensor数据转为numpy数据
    maxValue=array1.max()
    array1=array1*255/maxValue#normalize，将图像数据扩展到[0,255]
    mat=np.uint8(array1)#float32-->uint8
    print('mat_shape:',mat.shape)#mat_shape: (3, 982, 814)
    mat=mat.transpose(1,2,0)#mat_shape: (982, 814，3)
    print(mat.shape)
    cv2.imshow("img",mat)
    cv2.waitKey()


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
        
        
class Yolov3(nn.Module): #增加transform
    def __init__(self, num_classes=20, channels=3, device = 'cuda:0', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.device = device
        self.min_size = 384
        self.max_size = 512
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.transform = None
        self.backbone = backbone(num_classes=num_classes)
        self.feature_maps = []   
        self.outputs = [] 
        self.layers = nn.ModuleList()
        self.layers += [
            YoloBlock(1024, 512), Prediction(512, self.num_classes), DBL(512,256,1),
            YoloBlock(768, 256), Prediction(256, self.num_classes), DBL(256,128,1),
            YoloBlock(384, 128), Prediction(128, self.num_classes)
        ]
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, images):
        for img in images: # img[C,H,W]
            val = img.shape[-2:] # 获得[H,W]
            assert len(val) == 2  # 防止输入的是个一维向量

        # images是一个batch的图片组成的list，batch中的图片尺寸可能不一样。对于VOC2012数据集，大约在375*500
        self.transform = Transform(self.min_size, self.max_size, self.image_mean, self.image_std)
        images, targets = self.transform(images,None)  # 对图像进行预处理,将图像大小统一
        #输出尺寸为375*500，这是预设的参数
        
        backbone_modules = self.backbone.layers._modules
        x = images.tensors.to(self.device)
        # tensor2img(x)
        for name in backbone_modules:
            x = backbone_modules[name](x)
            print(x.shape)
            if name == '3' or name == '4':
                self.feature_maps.append(x)
        # layers_names = ['0','1','2','3','4','5','6','7']
        for name in range(8):
            # print(name)
            if name == 0 or name == 3 or name == 6:
                x = self.layers[name](x)
            elif name == 1 or name == 4 or name == 7:
                self.outputs.append(self.layers[name](x)) 
            elif name == 2 or name == 5:
                x = self.upsample(self.layers[name](x))
                # dim=1,[N,C,H,W]
                x = torch.cat([x,self.feature_maps[-1]],dim = 1)
                self.feature_maps.pop()
        return self.outputs
    # outputs[0,1,2]  outputs[0]:[N,C,H,W]
    # C = 85 = 3*(num_classes+4+1),[x,y,h,w,co,class]

