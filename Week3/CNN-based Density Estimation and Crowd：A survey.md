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