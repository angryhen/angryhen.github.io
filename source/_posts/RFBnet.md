---
title: RFBnet
date: 2019-01-02 16:10:19
tags: '论文解读'
categories: '目标检测'
keywords: '深度学习'
description: '对RFBNet的论文解读'
top_img: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-881c898b2e88c0fa.png
cover: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-881c898b2e88c0fa.png
---

论文地址：https://arxiv.org/pdf/1711.07767.pdf
官方源码（pytorch）：https://github.com/ruinmessi/RFBNet

### Abstract
&#160; &#160; &#160; &#160;主要说了目前表现好的目标检测主要基于较深的网络（例如Resnet，Inception），其缺点就是大量的计算成本，速度慢。而一些轻量级的网络速度较快，但检测的精度相对不高。作者提出了RFB模块，并将它添加到SSD的顶部，构建了RFBnet。

### Introduction
&#160; &#160; &#160; &#160;为了构建快速而强大的探测器，合理的替代方案是通过引入某些手工制作的机制来增强轻量级网络的特征表示，而不是一味地加深模型。

>Regarding current deep learning models, they commonly set RFs at the same
>size with a regular sampling grid on a feature map, which probably induces some
>loss in the feature discriminability as well as robustness
>对于当前的深度学习模型，它们通常将RFs设置为与特征图上常规抽样网格相同的大小,这可能会导致一些
>特征可辨性和鲁棒性的损失（这句不太懂具体的原理，后续再补充）

![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-8bd7834a73264b34.png)
>本文提出了一种新颖的模块，即 RFB，目的就是：**以加强从轻量级CNN模型中学到的深层特征，使它们有助于快速准确的探测器**

RFBnet 结构说明

![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-c1157b1c973418bb.png)
>1.RFB模块就是模拟人类视觉系统中RF的大小和离率，旨在增强轻量级CNN网络的深层功能
>2.提出基于RFB网络的检测器，通过用RFB替换SSD的顶部卷积层，显着的性能增益，同时仍然保持受控的计算成本
>3.RFBnet以实时处理速度在Pascal VOC和MS COCO上实现了最先进的结果，并通过将RFB链接到MobileNet来展示RFB的泛化能力

### Related Work
过

###  Method
#### Receptive Field Block
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-88f495367b49e55d.png)
RFB其实就是多分枝卷积块，其内部结构由两部分组成：
1.前一部分与inception一致，负责模拟多尺寸的pRF
2.后一部分再现了人类视觉中pRF与离心率的关系
下图给出了RFB及其对应的空间池区域图
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-881c898b2e88c0fa.png)
#### Multi-branch convolution layer
具体来说，首先，我们在每个分支中采用瓶颈结构，由1×1转换层组成，以减少特征映射中的通道数量加上n×n转换层。其次，我们用两个堆叠的3×3转换层替换5×5转换层，以减少参数和更深的非线性层。出于同样的原因，我们使用1×n加n×1转换层来代替原始的n×n转换层。最后，我们应用ResNet 和Inception-ResNet V2 的快捷方式设计。

#### Dilated pooling or convolution layer
也叫做astrous卷积层，**该结构的基本意图是生成更高分辨率的特征图，在具有更多上下文的更大区域捕获信息，同时保持相同数量的参数**。

>we exploit dilated convolution to simulate the impact of the eccentricities of pRFs in the human visual cortex
>我们利用空洞卷积来模拟pRF在人类视觉皮层中的离心率的影响

下图示出了多分支卷积层和扩张合并或卷积层的两种组合
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-6885b5eeff36d5c3.png)

### RFB Net Detection Architecture![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-fd503da087ef9183.png)

所提出的RFB网络探测器重用了SSD的多尺度和单级框架，其中RFB模块被嵌入以改善从轻量级主干提取的特征，使得探测器更准确且仍然足够快。 由于RFB的特性可以轻松集成到CNN中，我们可以尽可能地保留SSD架构。 主要的修改在于用RFB代替顶部卷积层
#### Lightweight backbone
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-0cf3abf7a555ba45.png)
使用与SSD中完全相同的骨干网络。 简而言之，它是在ILSVRC CLS-LOC数据集上预先训练的VGG16 ，其中fc6和fc7层被转换为具有子采样参数的卷积层，并且其pool5层从2×2-s2变至3×3-s1。 空洞卷积层用来填充空缺和所有dropout层，并移除fc8层。
#### RFB on multi-scale feature maps
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-7558f61b5bc11f4d.png)



保持相同的SSD级联结构，但具有相对较大分辨率的特征映射的卷积层被RFB模块取代。 在RFB的主要版本中，我们使用单一结构设置来模仿离心率的影响。 随着视觉图之间pRF大小和离心率的差异，我们相应地调整RFB的参数以形成RFB-s模块，其模拟浅人类视网膜图中较小的pRF，并将其置于conv4 3特征之后，如 由于其特征映射的分辨率太小 而无法应用具有大型内核（如5×5）的滤波器，因此保留了最后几个卷积层。
#### Training Settings
train主要遵循SSD，包括数据增强，硬负挖掘，默认框的比例和宽高比，以及损失函数（例如，用于定位的平滑L1损失和用于分类的softmax损失），同时我们稍微改变了我们的学习速率调度 更好地适应RFB。 更多细节在以下实验部分中给出。 使用MSRA方法初始化所有新的conv层。

后面主要是描述研究的成果，与其他网络的对比，就不多描述了，以后补充更多关于RFBnet的细节