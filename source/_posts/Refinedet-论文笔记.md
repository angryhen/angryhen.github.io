---
title: "Refinedet-论文笔记"
date: 2018-12-28 17:50:42
tags: '论文解读'
categories: '目标检测'
keywords: '算法'
description: 'Refinedet 是CVPR2018的一篇论文，本人水平不高，在目标检测方面接触不多，仅以此文记录学习过程，主要内容在于对原文paper的翻译以及一些借鉴与解读'
top_img: https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=551013663,1774317073&fm=26&gp=0.jpg
cover: https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=551013663,1774317073&fm=26&gp=0.jpg
---

2018.12.28 对于refinedet 网络部分的一些补充：
https://www.jianshu.com/p/19896a763d1f

# 前述

&#160; &#160; &#160; &#160;Refinedet 是CVPR2018的一篇论文，本人水平不高，在目标检测方面接触不多，仅以此文记录学习过程，主要内容在于对原文paper的翻译以及一些借鉴与解读。

论文地址：https://arxiv.org/pdf/1711.06897.pdf
# 摘要
![Abstract.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231656)

其中主要说了：
**1**.包括了两个模块:anchor refinement mododule (**ARM**),
　　　　　　　　object detection module(**ODM**)
　　前者过滤掉负anchor以减少分类器的搜索空间，并且粗略调整anchor的位置和大小为后续的回归器提供更好的初始化*(类似于RPN)*，后一个模块将精确的anchor作为输入从前者进一步改进回归并预测多类标签。
**2**.一个转移连接块:transfer connection block(**TCB**)
**3**.多任务的端到端训练
>*后面会对这些有详细的解析*
>*同时作者给出了源码：https://github.com/sfzhang15/RefineDet (caffe)*

# 1.Introduction
## one-stage and two-stage
&#160; &#160; &#160; &#160;前３段主要介绍了目前object detection的算法框架，其中
>*However, its detection accuracy is usually behind that of
>the two-stage approach, one of the main reasons being due
>to the class imbalance problem*
>译：**但是，它的检测精度通常落后于两阶段方法，其中一个主要原因是由于阶级不平衡问题**

作者的观点中，描述了two-stage methods：Faster R-CNN, R-FCN, and FPNd的３个优点：
>*(1)using two-stage structure with sampling heuristics to handle classimbalance; 
>**采用带抽样启发式的两阶段结构来处理类不平衡**;
>(2) using two-step cascade toregress the object box parameters; 
>**使用两步级联来回归对象框参数**
>(3) using two-stage features to describe the objects
>**使用两阶段特征来描述对象***

## Refinedet 架构
![Refinedet架构.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231724)

>*Specifically, it achieves 85.8% and 86.8% mAPs on VOC2007 and 2012, with VGG-16 network. Meanwhile, it outperforms the previously best published results from bothone-stage and two-stage approaches by achieving 41.8% AP4 on MS COCO test-dev with ResNet-101. In ad3The features in the ARM focus on
>distinguishing positive anchors from background. We design the TCB to transfer the features in the ARMto handle the more challenging tasks in the ODM, i.e., predict accurate object locations, sizes and multi-class labels. Based on the evaluation protocol in MS COCO [29], AP is the sindition, RefineDet is time efficient, i.e., it runs at 40.2 FPS and 24.1 FPS on a NVIDIA Titan X GPU with the input sizes 320 × 320 and 512 × 512 in inference.*
>这一段详细说明了Refinedet 在**VOC2007,2012上采用VGG16，在MS COCO 上采用Resnet-101**，检测的结果和时间效率都优于目前公布最好的结果。

主要做了３个贡献：
１.引入了一个新颖的一阶段框架用于对象检测，由两个互连模块组成，即ARM和ODM。 这导致了性能比两阶段方法更好，同时保持一阶段方法的效率。
 2 .为了确保有效性，我们设计TCB以转移ARM中的特征以处理更具挑战性的任务，即在ODM中预测准确的对象位置，大小和类标签。
 3 .RefineDet在通用物体检测上实现了最新的最新成果（即PASCAL VOC 2007，2012和MS COCO）

# 2. Related Work
主要介绍了传统的和目前的一些目标检测算法，过。

# 3.Network Architecture 网络架构
&#160; &#160; &#160; &#160;第一段大体讲述了Refinedet的机理，类似于ssd的一个前馈卷及神经网络

>*Similar to SSD , RefineDet is based on a feedforward convolutional network that produces a fixed number of bounding boxes and the scores indicating the presence of different classes of objects in those boxes, followed by the non-maximum suppression to produce the final result*

类似于SSD，RefineDet基于前馈卷积网络，它产生固定数量的边界框，分数表示在这些框中存在不同类别的对象，然后是非极大值抑制以产生最终结果

## ARM结构

>***ARM** is constructed by **removing the classification layers and adding some auxiliary structures** of **two base networks**(i.e., VGG-16 [43] and ResNet-101 [19] pretrained on ImageNet [37]) to meet our needs*

**ARM**是在两个基础网络上(预训练的**VGG-16**和**ResNet-101**)，通过**移除分类层和添加一些辅助的结构**，达到我们的需求

## ODM

>*The **ODM** is composed of the outputs of **TCBs** followed by the prediction layers (i.e.,the convolution layers with 3 × 3 kernel size), which generates the scores for object classes and shape offsets relative to the refined anchor box coordinates*

**ODM**是由跟随在预测层（生成分类对象的分数和相对形状偏移的refined anchor box 的笛卡尔坐标）后面的TCBs的输出组成

## Transfer Connection Block(TCB)
![TCB.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231742)

&#160; &#160; &#160; &#160; 为了建立ARM和ODM的联系，我们引入TCB来将ARM中的特征图转换到ODM中，这样ODM可以共享ARM的特征。值得注意的是，从ARM中，我们只在与anchors有联系的特征图上使用TCBs。
　　TCB的另一个功能是通过向传输的特征添加高级特征来集成大规模的上下文，以提高检测精度。 为了匹配它们之间的尺寸，我们使用反卷积操作来放大高级特征图并以元素方式对它们求和。 然后，我们在求和之后添加卷积层以确保检测特征的可辨性
![TCBS.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231750)

>该网络主要有三个特点 
>1）利用TCB模块进行类似FPN 的特征融合，提高低层语义信息，有利于小物体检测 
>2）两步级联回归，提升框的质量，在ARM模块中利用SSD二分类网络做PRN的工作，进行粗回归调整，在ODM模块中进行位置精调。 
>3）负样本过滤机制，文中在进行1：3的难例挖掘前先进行了负样本的过滤，当候选框的背景置信度高（大于0.99时)，直接舍去，不丢入分类器，这样能缓解样本不平衡问题，同时缩短检测速度。
>####Two-Step Cascaded Regression
>*Specifically, we associate n anchor boxes with each regularly divided cell on the feature map. The initial position of each anchor box relative to its corresponding cell is fixed. 
>At each feature map cell, we predict four offsets of the refined anchor boxes relative to the original tiled anchors and two confidence scores indicating the presence of foreground objects in those boxes. Thus, we can yield n refined anchor boxes at each feature map cell. *

具体来说，就是将n个anchor box与特征图上的每个规则划分的单元格相关联。 每个anchor box相对于其相应单元的初始位置是固定的。
在每个特征地图单元格中，我们预测相对于原始anchor的refined anchor的四个偏移量和两个置信度分数(表示这些框中存在前景物体)。 因此，我们可以在每个特征图单元格产生n个refined anchors。获得refined anchors后，我们将其传到ODM相应的特征图中，进一步生成对象类别和准确的对象位置、尺寸。ARM和ODM中相应的特征图具有相同的维度。我们计算refined anchors的c个类别分数和四个准确的偏移量，产生c + 4的输出以完成检测任务。此过程类似于SSD 中的默认框。但是，与SSD 不同，RefineDet使用两步策略，即ARM生成refined anchor boxes，ODM采取其作为输入进一步检测，因此检测结果更精准，特别适用于小物体。

## Negative Anchor Filtering

> *in training phase, for a refined anchor box, if its negative confidence is larger than a preset threshold θ (i.e., set θ = 0.99empirically), we will discard it in training the ODM*

在训练阶段，对于精确的锚箱，如果其负置信度大于预设阈值θ（即设定θ= 0.99经度），我们将在训练ODM时将其丢弃，对应上面第三个特点

# 4. Training and Inference
## Data Augmentation
简单的说就是通过数据增强使得模型更具鲁棒性，随机扩展并裁剪原始训练图像，随机光度失真和翻转生成训练样本。 Please refer to [ssd]http://www.cs.unc.edu/~wliu/papers/ssd.pdf for more details.

## Backbone Network
![Backbone Network.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231804)

骨干网络使用了在ILSVRC CLS-LOC上预先训练的VGG-16和ResNet-101，同时也可以在其他的预训练网络上working*(such as Inception V2 [22], Inception ResNet [44], and ResNeXt101 )*。

>*we convert fc6 and fc7 of VGG-16 to convolution layers conv fc6 and conv fc7 via subsampling parameters
>Meanwhile, to capture high-level information and drive object detection at multiple scales,we also add two extra convolution layers (i.e., conv6 1 and conv6 2) to the end of the truncated VGG-16 and one extra residual block (i.e., res6) to the end of the truncated ResNet101, respectively.*

通过下采样参数(应该是吧)将**VGG-16的fc6和fc7转换为卷积层conv fc6和conv fc7**,同时为了在多个尺度上捕获高级信息和驱动对象检测，还在**截断的VGG-16的末尾添加了两个额外的卷积层(即conv6_1和conv6_2)和一个额外的残余块(即res6)添加到截断的ResNet101的末尾。**
>*Since conv4 3 and conv5 3 have different feature scales compared to other layers, we use L2 normalization [31] to scale the feature norms in conv4 3 and conv5 3 to 10 and 8, then learn the scales during back propagation*
>**对conv4_3以及conv5_3添加了L2 normalization层，并分别设置scale为10和8，并在反向传播中学习scale上**

附上VGG16和ResNet-101的结构图
![VGG-16.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231821)
![ResNet-50.jpg]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231828)

####Anchors Design and Matching

![Anchors Design and Matching.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231840)

Anchor的设计跟SSD也是比较相似的，不同的是，这里只在4个feature layer上面提取Anchor，分别对应stride为（8，16，32，64），并且不同的feature layer匹配不同大小及尺寸的anchor，scale是stride的4倍即对应的检测尺度为，*以320为例子，对应的不同的layer检测的图像尺度为：[ 32， 64， 128， 256 ]，aspect ratio 有3个（0.5，1，2）*,同时，在训练期间阶段，我们确定之间的对应关系基于anchors和ground truth boxes的jaccard重叠率（IoU），并端到端地训练整个网络。具体来说，我们首先将每个ground truth boxes与具有最佳重叠分数的anchor boxes相匹配，然后匹配anchor重叠高于0.5的任何ground truth boxes。

## Hard Negative Mining
![Hard Negative Mining.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231907)

在匹配步骤之后，大部分anchor boxex都是负样本的，即使在ARM时过滤了很多，类似于SSD的做法，用hard negative mining来设定正负样本的比例（一般设定为1:3），负样本不是随机选的，而是根据box的分类**loss排序**来选的，按照指定比例选择**loss最高**的那些负样本即可

####Loss Function
这里由两部分组成：
**ARM**：每个anchor的二分类标签*binary class label*(object is or not)$L_b$和其位置与大小的回归$L_r$
　　---->之后，我们将具有小于阈值的负置信度的精确锚传递给ODM，以进一步预测对象类别和准确的对象位置和大小。
**ODM**：多酚类multi-class classification损失$L_m$和回归损失$L_r$

需要注意的是:
　　　虽然本文大致上是RPN网络和SSD的结合，但是在Faster R-CNN算法中RPN网络和检测网络的训练可以分开也可以end to end，而这里的训练方
式就纯粹是end to end了，ARM和ODM两个部分的损失函数都是**一起向前传递**的。 
![Loss.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231932)

>$p_i$和$x_i$代表ARM中anchor分类的置信度和回归的坐标
>$c_i$ 和 $t_i$代表ODM中refined anchor分类的置信度和坐标回归
>$N_{arm}$和$N_{odm}$ 代表batch中的正样本数
>$L_b$ 代表二分类的交叉熵loss，作判断是否有object
>$L_m$代表softmax loss，多分类损失
>$L_r$ 代表smooth L1 loss，回归损失（应该和faster-rcnn类似吧）
>$l^*_i$  代表第$i$个anchor的ground truth的类别
>**[$l^*_i \geq 1$]**  表示如果negative confidence大于一个阈值θ，那么返回1，否则返回0
>　　　　也就是说当条件满足的时候输出为1，[$l^*_i \geq 1$]代表的是正样本，所以[$l^*_i \geq 1$]代表的是只
>　　　　有正样本才会去计算坐标回归，负样本不计算。
>$g^*_i$   代表第$i$个anchor的ground truth位置和大小

需要注意的是下面这一点：
![notably.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615231957)

## Optimization
![Optimization.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615232006)

 VGG-16：新添加的卷积层(onv6_1和conv6_2)，用xavier初始化参数
ResNet-101：新添加的residual block, 采用均值为0，方差为0.01的高斯分布进行初始化

>default batch size = 32
>momentum = 0.9 (收敛快)
>weight decay = 0.0005  (正则项，防止过拟合)
>learning rate = $10^{-3}$

## Inference
在inference阶段，ARM先过滤掉负置信度大于阈值$\theta$的anchor，然后refine剩余anchor的位置与大小，将refined anchor传入ODM模块进行分类，每张图像取得分高的400个图像。最终 应用NMS，jaccard重叠率限定为0.45，保证最终得到200个高分的检测结果作为最终的结果。

# 读后：

>1.感觉ARM就是和RPN的功能相差无几，ARM由多层不同尺度的特征输出，而RPN只有一个
>2.ODM接收来自ARM的refined anchor，类似于RPN的proposal，浅层的feature map 融合了高层feature map的信息，后预测bbox是基于每层feature map（每个蓝色矩形块）进行，最后将各层结果再整合到一起。此处提高了对小物体的检测
>3.TCB是将不同层次的ARM特征转化为ODM，它这里有一个回传的操作，将高层次的特征通过去卷机操作（实际是一种转置卷积），使特征图之间的尺寸匹配，然后与低层次的特征相加。

## 具体网络结构是怎么构建的呢?
以特征提取网络为ResNet101，输入图像大小为320为例，在Anchor Refinement Module部分的4个灰色矩形块（feature map）的size分别是40*40,20*20,10*10,5*5，其中前三个是ResNet101网络本身的输出层，最后5*5输出是另外添加的一个residual block。有了特征提取的主网络后，就要开始做融合层操作了，首先是5*5的feature map经过一个transfer connection block得到对应大小的蓝色矩形块（P6）,transfer connection block后面会介绍 ，对于生成P6的这条支路而言只是3个卷积层而已。接着基于10*10的灰色矩形块（feature map）经过transfer connection block得到对应大小的蓝色矩形块（P5），此处的transfer connection block相比P6增加了反卷积支路，反卷积支路的输入来自于生成P6的中间层输出。P4和P3的生成与P5同理。

作者的backbone采用VGG16（the conv4_3, conv5_3, conv fc7, and conv6_2 feature layers）和Resnet101(res3b3, res4b22, res5c, and res6)作为ARM的四个蓝色框
参考源码自定义ARM输出层为：
![output.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615232048)

>*需要注意：fc6和fc7采用conv层替代，conv6 的输入为32×32，采用dilated方式；conv7采用11卷积，输出32×32，同时增加5个类似ssd的conv输出层*

在此附上SSD的结构，易于对比：
![SSD.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/20200615232105)