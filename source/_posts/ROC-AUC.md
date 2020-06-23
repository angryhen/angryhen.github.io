---
title: 评价指标（二）ROC和AUC
date: 2019-10-07 16:22:57
tags: 
  -  评价准则
categories: ' 评价准则'
keywords: 'roc, AUC'
description: '继上篇文章[评价指标（一）精确率，召回率，F1-score]([https://www.jianshu.com/p/b29bfbf05ecf](https://www.jianshu.com/p/b29bfbf05ecf)
)，除了上述三个指标，这次深入讲述何为**ROC**与**AUC**，以及它们是如何工作的。'
top_img: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-ec36e93fd88f5cf1.png
cover: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-ec36e93fd88f5cf1.png
---

## 前言

&emsp;&emsp;继上篇文章[评价指标（一）精确率，召回率，F1-score]([https://www.jianshu.com/p/b29bfbf05ecf](https://www.jianshu.com/p/b29bfbf05ecf)
)，除了上述三个指标，这次深入讲述何为**ROC**与**AUC**，以及它们是如何工作的。

## ROC

**&emsp;&emsp;ROC(Receiver Operating Characteristic)** 翻译过来就是“**受试者工作特征**”，源于二战中用于敌机检测的信号雷达分析术，后来引入到机器学习的领域，当然，前提还是针对二分类问题。
首先我们得到一个混淆矩阵
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-ec36e93fd88f5cf1.png)
对于ROC，
横轴为**FPR**，预测为正中但实际为负/实际负样本数的比例，对应就是$\frac{FP}{\left(FP+TN\right)}$
纵轴为**TPR**，预测为正中实际也为正/实际正样本数的比例，对应就是$\frac{TP}{\left(TP+FN\right)}$，其实就是Recall
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-588895dfc5189647.png)
描绘ROC曲线的图成为“ROC图”，描述了TPR和FPR之间的相对平衡，上图显示了A~E五个分类器的ROC图

>有几个点要注意，对于一般而言
>（0，0） 代表阈值为1，全部判定为负类
>（1，1） 代表阈值为0，无条件判定为正类
>（0，1） 理想模型，模型预测百分比正确
>而在（0.5， 0.5）我们可以认为模型在瞎猜

很明显，越靠近D点，模型性能就越好，A相对于B更保守，事实上很多数据都是由大量的负类主导，由此看来或许A性能B的好；出现在右下角的任何分类器效果比随机猜测都差（如E），一般而言这块区域是空的。

### 如何得到ROC曲线

给定一个测试机，我们可以通过阈值threshold：大于threshold判定为正，反之为负
以论文的例子展开：测试集一共20个样本，**Class**为其正式的标签，**Score**为模型预测为正类的概率，我们根据**Score**对其排序
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-6000373e9f74a531.png)
从高到低，依次**Score**值作为threshold，通过计算可以得到20组的（FPR, TPR），即得到一条ROC曲线。
例如：以**Score**=0.55作为threshold，可以得到混淆矩阵
![threshold=0.55]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-b6f9beb439ba6816.png)

>那么TPR = 4/(4+6) = 0.4
>FPR = 1/(1+9) = 0.1
>最后得到坐标（0.1，0.4）

最后结果如下图：
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-fb7487788da58084.png)
现实任务中通常是利用有限的测试样本绘制ROC图，如果想得到平滑的曲线，可以通过增加测试样本去拟合，但一般我们不会选择这么做。
若分类器A的ROC曲线被另一个分类器B的ROC曲线完全覆盖（如下图），则B的性能绝对优于A
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-e7fbc893b989ac3a.png)
若分类器A的ROC曲线和分类器B的ROC曲线发生交叉（如下图），则难以判定孰优孰劣，此时如果一定要对比，就可以用AUC进行判断
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-0a2f2faa12a9ec78.png)

## AUC

**AUC(Area Under ROC Curve)**，就是ROC曲线下面的面积
假定ROC曲线有坐标为$\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}$的点连续链接而形成，则AUC可估算为：
$$AUC\;=\;\frac12\sum_{i=1}^{m-1}(x_{i+1}-x_i)\ast(y_i-y_{i+1})$$

## END

ROC曲线对于分类器是个二位的描述，简单来说我们希望能通过AUC这样单个标量去衡量模型的性能，且又不能像Recision，Recall，F1这类有时候无法正确解释的指标