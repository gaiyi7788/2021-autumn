import torch
import torch.nn as nn
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss 
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size = 1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__() 
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3,padding=1),
            # 每个cell对应三种anchor
            CNNBlock(2*in_channels,3*(num_classes+5),bn_act=False,kernel_size=1) # [po,x,y,w,h]
        )
        self.num_classes = num_classes
    
    def forward(self,x):
        return (
            self.pred(x)
            .reshape(x.shape[0],3,self.num_classes+5,x.shape[2],x.shape[3])
            .permute(0,1,3,4,2)
        )
        #    N x (anchor_nums) x (w) x (h) x (5+num_classes)
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__() 
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self,x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            print(x.shape)
            if isinstance(layer,ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            # 观察到重复了8次的ResidualBlock才会与后面进行concat
            if isinstance(layer, ResidualBlock) and layer.num_repeats==8:
                route_connections.append(x)
            # 上采样完成以后就是拼接过程，取最后一个加进来的进行拼接，然后pop()使前一个变成最后一个
            elif isinstance(layer, nn.Upsample): 
                x = torch.cat([x,route_connections[-1]],dim = 1)
                route_connections.pop()
        return outputs
    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for Module in config:
            if isinstance(Module,tuple):
                out_channels ,kernel_size, stride = Module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride = stride,
                        padding=1 if kernel_size == 3 else 0 #1*1卷积不用padding，3*3需要
                    )
                )
                in_channels = out_channels
            elif isinstance(Module, list):
                num_repeats = Module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(Module, str):
                if Module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size = 1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels//2
                    
                elif Module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels*3 #上采样的时候同时想要连接
        return layers
    
    
if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2,3,IMAGE_SIZE,IMAGE_SIZE)) #(N,C,H,W)
    out = model(x)
    assert model(x)[0].shape == (2,3,IMAGE_SIZE//32,IMAGE_SIZE//32,num_classes+5)
    assert model(x)[1].shape == (2,3,IMAGE_SIZE//16,IMAGE_SIZE//16,num_classes+5)
    assert model(x)[2].shape == (2,3,IMAGE_SIZE//8,IMAGE_SIZE//8,num_classes+5)
    print("success!")