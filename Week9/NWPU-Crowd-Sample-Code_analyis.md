[TOC]

## 工程主干

```
NWPU-Crowd-Sample-Code
├── config.py
├── datasets
│   ├── basedataset.py
│   ├── common.py
│   ├── get_density_map_gaussian.m
│   ├── get_dot_map.m
│   ├── __init__.py
│   ├── preapre_NWPU.m
│   ├── preapre_NWPU.mlx
│   ├── __pycache__
│   └── setting
├── exp
│   ├── code/
│   ├── 10-25_16-12_NWPU_CSRNet_1e-05.txt
│   ├── all_ep_1_mae_265.2_mse_709.2_nae_4.734.pth
│   ├── all_ep_6_mae_182.1_mse_624.6_nae_1.890.pth
│   ├── events.out.tfevents.1635149647.ubuntu
│   └── latest_state.pth
├── __init__.py
├── LICENSE
├── misc
│   ├── cal_mean.py
│   ├── dot_ops.py
│   ├── evaluation_code.py
│   ├── __init__.py
│   ├── layer.py
│   ├── __pycache__
│   ├── transforms.py
│   └── utils.py
├── models
│   ├── CC.py
│   ├── counters
│   │   ├── CANNet.py
│   │   ├── CSRNet.py
│   │   ├── __init__.py
│   │   ├── MCNN.py
│   │   ├── __pycache__
│   │   ├── Res101_SFCN.py
│   │   ├── SCAR.py
│   │   └── VGG.py
│   ├── __init__.py
│   └── __pycache__
├── NWPU-Crowd
├── qualitycc.py
├── README.md
├── requirements.txt
├── saved_exp_para
│   ├── CANNet
│   ├── CSRNet
│   │   ├── config.py
│   │   └──  NWPU.py
│   ├── MCNN
│   ├── Res_SFCN
│   ├── SCAR
│   └── VGG
├── test.py
├── trainer.py
├── train.py
└── validation.py
```

## NWPU-Crowd数据集介绍

```
 -- NWPU-Crowd
        |-- images
        |   |-- 0001.jpg
        |   |-- 0002.jpg
        |   |-- ...
        |   |-- 5109.jpg
        |-- jsons
        |   |-- 0001.json
        |   |-- 0002.json
        |   |-- ...
        |   |-- 3609.json
        |-- mats
        |   |-- 0001.mat
        |   |-- 0002.mat
        |   |-- ...
        |   |-- 3609.mat
        |-- min_576x768_mod16_2048
        |   |-- den
        |   |-- dot
        |      |-- 0001.png
        |      |-- 0002.png
        |      |-- ...
        |      |-- 3609.png
        |   |-- img
        |   	|-- 0001.jpg
        |   	|-- 0002.jpg
        |   	|-- ...
        |   	|-- 5109.jpg
        |   |-- txt_list
        |       |-- train.txt
        |       |-- val.txt
        |       |-- test.txt
        |-- train.txt
        |-- val.txt
        |-- test.txt
        |-- readme.md
```

其中，min_576x768_mod16_2048文件夹下的内容是通过调用 datasets/preapre_NWPU.m生成的，里面只有img和dot文件是必要的，然后将原本NWPU-Crowd文件夹下的train.txt, val.txt, test.txt三个文件方法哦 txt_list/文件夹下。

- img文件是经过了预处理，size为16的倍数
- dot文件也根据resize的的img的尺寸进行了调整，大小相同，单通道图，每个坐标的数值对应该像素位置的人头数（因为原标注文件的人头位置为亚像素值，在matlab文件中进行了取整合并，因此一个位置可能有多个人头，后续通过高斯核扩散可以消除这种取整结果带来的影响）

## config.py

在 `saved_exp_para`文件夹中，针对不同的网络，有config.py和NWPU.py两个文件，在跑不同网络时记得把config.py文件换成对应的config文件，然后NWPU.py中记得更新数据集的位置。

## misc

- 用`dot_ops.py`的class Gaussian(nn.Module) 和 `layer.py`中的class Gaussianlayer(nn.Module) 共同根据数据集中的dot生成density_map
