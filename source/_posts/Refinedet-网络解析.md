---
title: "Refinedet 网络解析"
date: 2018-12-30 18:10:23
tags: '论文解读'
categories: '目标检测'
keywords: '算法'
description: '接着上一篇refinedet，瓷片主要解析一下网络结构'
top_img: https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=551013663,1774317073&fm=26&gp=0.jpg
cover: https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=551013663,1774317073&fm=26&gp=0.jpg
---

# refinedet 论文笔记：
https://www.jianshu.com/p/5504f4188d52

&#160; &#160; &#160; &#160;纸上学来终觉浅，继上次读完Refinedet论文后，理论上理解了其中的原理，后面主要花了些功夫阅读了源码，理解整个网络的数据流程。
![image.png](https://upload-images.jianshu.io/upload_images/15147802-3b4727aa40bc1ff7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&#160; &#160; &#160; &#160;照例先上论文中的架构图，在此我对里面做了一些标注
![image.png](https://upload-images.jianshu.io/upload_images/15147802-136ece825ad8184b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
上面是ARM里4个特征图的Size，左边为320x320，右边为512x512的原图像输入

网络改动的地方，是在VGG16的"pool5"之后，添加了**conv fc6，conv fc7，conv6_1以及conv6_2**，还有**ODM的[P6, P5, P4, P3]**, 其中
# ARM部分
## --conv fc6 

>通过在"pool5"上做了**atrous convolution，kernel_size = 3x3,stride = 3 , channel = 1024,其中stride=3指在map上两两相隔3-1=2 个步长**，输出的大小不变，channel为1024，见下图即可明白：

>1.应该同样存在信息损失的情况，传统conv如果stride为1，则会有一部分重叠，而dilation conv极大减少这点
>2.主要为了在不损失信息的情况下增大感受野，而扩大conv的尺寸也可以，但参数会变得更多，而且conv的增大和感受野的增大是线性，但dilation conv和感受野是指数增长

![image.png](https://upload-images.jianshu.io/upload_images/15147802-503364898cdbcf98.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## --conv fc7

>在"conv fc6"之后做了**kernel_size = 1x1,stride = 1 , channel = 1024**的卷积操作，形状与conv fc6 保持一致
>####--conv6_1
>继"conv fc7"之后，做了**kernel_size = 1x1,stride = 1 , channel = 256**的卷积操作，size不变，channel为256
>####--conv6_2
>"conv6_1"之后，**kernel_size = 3x3,stride = 2 , channel = 512**的卷积操作，也就是说把尺寸缩小一半了，channel变成512



# ODM部分
ODM部分主要通过TCB模块转换而成，回顾一下TCB的流程：
![image.png](https://upload-images.jianshu.io/upload_images/15147802-274d3156958ccf80.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Input （上方的箭头）为**ARM[conv 4_3，conv5_3，conv fc7，conv6_2]**，输出分别对应了**ODM[P3, P4, P5, P6]**
**deconv**这一步，传入的实际是[P4 , P5 , P6]，
根据网络的架构图，我们不难看出，**P6** 是 首先生成的，而conv6_2的输出已经是最高的feature map所以在conv6_2  到 P6 的TCB，没有deconv这个操作（如下图所示）
![image.png](https://upload-images.jianshu.io/upload_images/15147802-4d9fe9a328376c28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## -- P6

>因为是最高层的输出，所以没有deconv操作，自然也就没Elte sum，实际就是feature map经过3个**kernel_size = 3x3,stride =1 , channel = 256，以及relu**的卷积操作，size不变，channel 为 256

## -- P5

>与"P6"不同，TCB 有了deconv的输入，就是P6。*(P3,P4同理)*
>"conv fc7" 输出的feature map **先经过一个 conv-relu和一个 conv 之后，再和 deconv后的P6 进行eltw_sum操作，而后接了一个relu，最后就是一个con_relu操作**，从而得到P5

**值得一提的是，TCB里的deconv，是把size加倍**

## -- P4 P3

>原理一样，看P5和 P6 即可明白



**有一个细节就是当conv4_3，conv5_3层在做anchor 的预测亦或者是做TCB操作的时候为了防止反向传播的剃度过大导致loss输出nan，这两层会经过一个L2normlization操作然后分别扩大常量倍数，scale的值分别为10和8**

ARM和ODM的feature map之后，执行了类似的卷积操作。
>**ARM：**1.**kernel_size = 3x3, stride = 1 ,channel = num_anchor✖4（坐标回归）**
>&#160; &#160; &#160; &#160;&#160;&#160;&#160;&#160; 2.**kernel_size = 3x3, stride = 1 ,channel = num_anchor✖2（判断前后景）**

>**ODM：**1.**kernel_size = 3x3, stride = 1 ,channel = num_anchor✖4（坐标回归）**
>&#160; &#160; &#160; &#160;&#160;&#160;&#160;&#160; 2.**kernel_size = 3x3, stride = 1 ,channel = num_anchor✖num_cls（分类）**

## 关于anchor 
正样本：选取IOU最大的一个anchor，以及剩下的IO>0.5的作为所有正样本
负样本：

>这里开始对论文的理解不太深，不知道是自己理解问题还是作者表达不太清楚，现在可以可以总结出：
>1.过滤掉所有负置信度>0.99的anchor，换而言之，就是去掉很**明显就能判断出是背景的anchor**，如果保留了这些，模型能很快而且轻松的学到把anchor判断为背景，这样反而是不好的，学习过程过于简单而且快，模型的理解能力会很差。
>2.相反，我们需要的是误判为物品的anchor，分数越高，证明偏差越大，把这些负样本送进网络训练，告知模型难以判断的错误，能学习到更复杂的情况和更好的效果，总而言之就是让模型尽可能学到更多。
>3.貌似在2的时候，ARM并没有把低分的score（误判的）过滤掉，而是一起送进ODM，最后ODM再结合loss计算把这些过滤掉

总体来说，ARM中先第一次调整anchor的位置、尺度，使之为ODM提供修正后的anchor；整体操作方式与RPN类似，在参与预测的feature map每个位置上，密集采样 n 个anchors，每个anchor的尺度、长宽比是预定义好的，位置也是相对固定的；ARM就相当于RPN操作，为每个anchor预测其相对位置偏置（relative offsets，也即，对anchor原始位置的相对位移），并预测每个anchor的objectness二分类得分，那么最终就得到了 n 个调整后的anchor，当然了，并不是所有anchor都被判定为包含目标，ARM就只需要筛选判别为目标的anchor，走下一步ODM的流程；
得到ARM调整的anchor之后，利用TCB得到anchor在feature map上新的特征，并在ODM上进一步实施类似SSD的操作，预测bbox的位置、大小、具体类别等；操作就跟SSD很像了，ARM、ODM上feature map的尺度一致，对ARM中每个判定为object的1-stage refined anchor，进一步输出 C 类得分 + 2-stage refined anchor（也即4个坐标offsets），输出也与SSD保持一致，每个anchor对应C + 4维的输出；

更多细节的东西以后再更