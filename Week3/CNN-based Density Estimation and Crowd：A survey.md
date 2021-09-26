[TOC]

# CNN-based Density Estimation and Crowd Counting: A Survey

## Abstract

In this paper, we have surveyed over 220 works to comprehensively and systematically study the crowd counting models, mainly CNN-based density map estimation methods. Finally, according to the **evaluation metrics**, we select the **top three performers on their crowd counting datasets and analyze their merits and drawbacks.**

在本文中，我们调查了 220 多个作品，以全面系统地研究人群计数模型，主要是基于 CNN 的密度图估计方法。 最后，根据**evaluation metrics**，我们选择在他们的人群计数数据集上表现最好的前三名，并分析他们的优缺点。

## I. INTRODUCTION

人群计数对于社会安全和控制管理起着不可或缺的作用。

虽然不同的任务有其独特的属性，但存在**结构特征和分布模式等共同特征**。 幸运的是，人群计数技术可以通过特定工具扩展到其他一些领域。

在本文中，我们希望通过对人群计数任务的深入挖掘，为其他任务提供合理的解决方案，尤其是基于 CNN 的密度估计和人群计数模型。

### A. Related Works and Scope

人群计数的各种方法主要分为四类：基于检测、基于回归、密度估计和最近基于CNN的密度估计方法

- detection-based：当在极密集的人群中遇到遮挡和背景杂乱的情况时，他们会呈现出不令人满意的结果。
- regression-based： directly learn the mapping from an **image patch** to the count. 它们通常首先提取全局特征（纹理、渐变、边缘特征），或局部特征(SIFT，LBP，HOG，GLCM)。然后是一些回归技术，如线性回归和高斯混合回归用来学习到人群计数的映射函数。
- detection-based和regression-based方法都忽略了空间信息。
- density estimation based：通过学习局部特征和相应的密度映射之间的线性映射。提出了一种非线性映射。但是只考虑了手工方法提取低级信息。
- CNN-based：全卷积网络FCN已成为密度估计和人群计数的主流网络架构。

我们将现有的方法分为网络架构、监督形式、跨场景或多领域的影响等几类。

### B. Related previous reviews and surveys

不同于以往关注手工制作特征或原始CNN的调查不同，我们的工作系统和全面地回顾了基于CNN的密度估计人群计数方法。

### C. Contributions of this paper

- 从各个方面进行全面和系统的概述
- 基于属性的性能分析
- 开放式问题和未来的发展方向

## II. TAXONOMY FOR CROWD COUNTING

### A. Representative network architectures for crowd counting

#### 1）Basic CNN

- 该网络体系结构采用了基本的CNN层，即卷积层、池化层、唯一的完全连接层，而不需要额外的特征信息。它们通常是 参与了使用CNN进行密度估计和人群计数的初步工作。

- 由于没有提供额外的特性信息，基本的CNN简单且易于实现，但通常执行精度较低。

####  2）Multi-column

这些网络架构通常采用不同的column来捕获对应于不同感受野的多尺度信息，这为人群计数带来了优异的性能。

##### MCNN

建立了一个多列卷积神经网络(MCNN)，并提出了用于图像分类的多列深度神经网络。

采用估计density map的方式而不是直接计总数，是因为density map包含了更多的空间信息，而且学习器能更好的适用不同size的head

- Input: The MCNN is the image

- Output:  A crowd density map whose integral gives the overall crowd count.

<img src="https://gitee.com/gaiyi7788/Typora_pictures/raw/master/img/20210926152657.png" alt="image-20210919130236360" style="zoom: 33%;" />

我们假设每个人的头部周围比较均匀，头部和它在图像中的 $k$ 个邻域的平均距离给出几何变形的合理估计，因此根据图像中每个人的头部大小确定传播参数 $σ$。我们发现人头大小通常与拥挤场景中两个相邻人的中心之间的距离有关，作为一种折中，对于那些拥挤场景的密度图，我们建议根据每个人与邻域的平均距离来自适应地确定每个人的传播参数.
$$
H(x) = \sum^N_{i=1}δ(x-xi) \\
F(x) = H(x) ∗ G_σ(x),\hspace{1em} with\hspace{0.5em} \sigma_i=\beta \overline{d^i}
$$

- 一个标记有N个head的图像表示为$H(x)$
- $G_{\sigma}(x)$是以$\sigma$为参数的高斯卷积核，$\overline{d^i}$是由k近邻方法计算的平均欧式距离，${\overline {d^i}}= \frac 1 m \sum_{j=1}^m d^i_j$
- $F(x)$是密度图

$$
L(\theta) = \frac{1}{2N}\sum^N_{i=1}||F(X_i;\theta)-F_i||^2_2
$$

- 其中Θ是MCNN中的一组可学习参数。
- N是训练图像的数量。
- Xi是输入图像，Fi是图像Xi的地面真实密度图。
- F(Xi; Θ)代表MCNN生成的估计密度图，用样本Xi的Θ参数化。
- L是估计的密度图和地面真实的密度图之间的损失。

##### Hydra-CNN

Hydra CNN学习了一个多尺度非线性回归模型，该模型使用在多个尺度上提取的图像斑块金字塔来进行最终的密度预测。

![image-20210919161625907](https://gitee.com/gaiyi7788/Typora_pictures/raw/master/img/20210926152732.png)

![image-20210919161453210](https://gitee.com/gaiyi7788/Typora_pictures/raw/master/img/20210926152739.png)
$$
D_I(p) = \sum_{\mu \in A_I} N(p;\mu,\Sigma)
$$
$D_I(p)$是密度图，$A_I$是为图像$I$标注的二维点集，$N(p;\mu,Σ)$表示一个归一化的二维高斯函数的求值，使用平均µ和各向同性协方差矩阵Σ，在p定义的像素位置进行求值。

对$D_I$积分可以得到总对象计数$N_I$：
$$
N_I = \sum_{p \in I} D_I(p)
$$

- Input: An image patch *P* 

- Output:  density map prediction<img src="CNN-based Density Estimation and Crowd：A survey.assets/image-20210919161520296.png" alt="image-20210919161520296" style="zoom:30%;" />，Ω是CNN模型的参数集
- loss function：<img src="CNN-based Density Estimation and Crowd：A survey.assets/image-20210919161728684.png" alt="image-20210919161728684" style="zoom: 40%;" />

##### CrowdNet

CrowdNet在不同的column上结合了浅层（shallow）和深度（deep）网络，其中浅层捕获大规模变化对应的低层特征，深度捕获高层特征语义信息。

<img src="https://gitee.com/gaiyi7788/Typora_pictures/raw/master/img/20210926152747.png" alt="image-20210920213237812" style="zoom:67%;" />

- Deep Network采用VGG-16为基本架构，但是第四个max-pooling层的stride变为1，去掉最后一个max-pooling层，将尺度变化由1/32变为1/8 。
- Shallow Network采用三个卷积层，卷积核个数均为24，为了确保没有由于max-pooling而造成的计数损失，在浅层网络中使用average-pooling。
- concat深层和浅层网络（axis=1），然后用1x1卷积进行信息融合，**在得到输出图像后采用双线性插值对输出密度图进行上采样，得到与输入图像相同的size。**（前面很多文章采用的都是对输入图像进行下采样保证与密度图的size相同）（我个人觉得采用双线性插值的方法得到的误差应该更大啊）
- 文章还提到了data augmentation，通过在单张图像构成的图像金字塔上获得大小均为225x225的patches，作为input。

##### Switching CNN

