---
title: "Batch normalization理解"
date: 2019-01-11 16:22:30
tags: '论文解读'
categories: '目标检测'
keywords: '深度学习'
description: '理解BN的具体原理'
top_img: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-d2735b465ecdc0fd.png
cover: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-d2735b465ecdc0fd.png
---

## 正文

&#160; &#160; &#160; &#160;在学习源码的过程中，发现在搭建网络架构的时候，经常会用到bn算法（即batch_normalization，批标准化），所以有必要深入去研究其中的奥妙。bn算法的提出在2015年的论文[《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》](https://arxiv.org/pdf/1502.03167v3.pdf)。
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-f2a112c5634134a0.png)


&#160; &#160; &#160; &#160;正如论文开始所说：由于训练过程中各层输入的分布随着前几层参数的变化而变化，使得训练深度神经网络变得复杂。这通过要求**较低的学习速率**和**仔细的参数**初始化来减慢训练，并且使得训练具有饱和非线性的模型变得非常困难。我们将这种现象称为 **internal covariate shift**。显而易见，为了解决这个问题，作者提出了bn，所以前提，我们得先理解何为internal covariate shift。

>将在训练过程中深度网络内部节点分布的变化称为internal covariate shift。这个internal可以看作隐层
>作者以sigmoid为例，z = g(wu+b)，其中u为输入层，w和b为权重矩阵和偏移向量（即网络层里需要学习的参数）
>![](https://upload-images.jianshu.io/upload_images/15147802-e883ff95d556c16d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)随着|x|的增加，g'(x)会趋向于0，这意味着，在所有维数上，除了那些绝对值较小的 x=wu+b ，下降到 u 的梯度将消失，模型将缓慢训练。

![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-614891fd65c2b567.png)
>x 受 W，b 和前面所有层的参数的影响，在训练过程中这些参数的变化可能会将 x 的许多维度移入非线性的饱和状态并减慢收敛速度。这种影响随着网络深度的增加而放大。


我们都知道在train网络之前，会对数据进行归一化处理，为的是保持训练和测试数据的分布相同，而在神经网络内部，每一层我们都需要有输出和输出，除了对原始数据的标准化处理，在经过网络每一层计算后的数据，它们的分布是不同的。网络的训练，需要去学习适应不同的数据分布，明显造成的后果就是收敛慢，效果不佳。
另一方面，网络前面的参数变更，会随着网络的深度，其影响不断累积增大，所以说只要有某一层的数据分布发生变化，后面层的数据输入分布也会不同，结合前面说的，为了解决中间层数据分布改变的情况。

**总的来说，bn的操作很简单，也很容易理解。就是在网络的每一层输入之前，做了一个归一化处理，就是作用于（wu+b），即bn(wu+b)，然后再接激活函数（非线性映射）。而且，很多论文的代码里，bn算作了独立的一层。**

公式如下：
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-03e071dd454bbe82.png)

E[x]为均值，sqrt（var）为标准差，然后加上scale和shift两个可训练的变量
而Batch Normalization可使各隐藏层输入的均值和方差为任意值。实际上，从激活函数的角度来说，如果各隐藏层的输入均值在靠近0的区域即处于激活函数的线性区域，这样不利于训练好的非线性神经网络，得到的模型效果也不会太好。这也解释了为什么需要用 γ 和 β 来做进一步处理
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-26f88a03f6fdfec0.png)
y(k)就是经过bn处理后的输出了，论文里提到，当![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-61c4aabf51a959c2.png)
可以恢复原始的激活，也就是这一层所学到的原始特征
对于批处理的推理如下，整个思路也很清晰：
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-d2735b465ecdc0fd.png)

到了这里，其实脑子还是很懵，一知半解，还是属于抽象的理解，对于其中还是很多疑惑
想要更好的理解，还是得回到数学的层面去看

一般来说，如果模型的输入特征不相关且满足标准正态分布时，模型的表现一般较好。
在训练神经网络模型时，我们可以事先将特征去相关并使得它们满足一个比较好的分布，
这样，模型的第一层网络一般都会有一个比较好的输入特征，
但是随着模型的层数加深，网络的非线性变换使得每一层的结果变得相关了，且不再满足分布。
甚至，这些隐藏层的特征分布或许已经发生了偏移。

![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-7ff841dabc6210c2.png)

在经过激活层之前，就是x = wu+b，也就是激活函数的输入，随着网络加深，x的会逐渐向两端靠拢（红色箭头的地方），那么会造成什么后果，我们可以看下这两个函数的导数:
Sigmoid' = sigmoid*（1-sigmoid）
Tanh' = 1-tanh^2
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-e23b66afc20159a3.png)

不难看出，在函数的两侧梯度变化趋向于0，且变化很慢，这会导致在Back propagation的时候梯度消失，也就是说收敛越来越慢。 在这里感觉也可以理解为:
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-c33c8df07949e572.png)
至于为什么要这么做最后的公式（scale，offset），论文里有提到
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-12a9f31d719b97fc.png)

>简单地对图层的每个输入进行规范化可能会更改图层可以表示的内容。
>例如，正则化一个 sigmoid 的输入将限制他们非线性的线性状态
>作者也没具体说明，在我也是不太明白其原理，只能理解为上面的变换会改变原来学习到的特征分布，因此加入了可学习的γ和β，为什么是可学习的，感觉应该是让网络自己找到一个在正态变换后不破坏原特征分布的平衡状态。

好吧，很玄，希望有小伙伴能给我解答一下。

## 测试

&#160; &#160; &#160; &#160;到这里，大致的原理就这样了，至于后面如何反向传播，以及推理过程中  均值mean 和 方差var 的设置，就不再写下去，网上也很多解读的资源。
理解了理论之后，结合实战操作才有意思，先附一张图.
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-0d0db612d9ff1ff9.png)

>在cnn中，batch_normalization就是取同一个channel上所有批次做处理，粗略画了这个示意图
>代表batch = 3，channel = 2 ， W和H = 2

下面用了numpy，pytorch以及tensorflow的函数计算batch_normalization
先看一下pytorch的函数以及描述
### nn.batchnorm2d   /  tf.layers.batch_normalization
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-1cc6034bb3437b3b.png)
最需要注意的是pytorch和tensorflow 的公式不太一样，所以结果会有稍微差异，在测试的时候，我们只需要把其他因素统一就好：
![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-95919bf46f444833.png)

```python
# 导库
import numpy as np
import tensorflow as tf
import torch.nn as nn
```
```python
# 创建一个随机矩阵
test1 = np.random.rand(4,3,2,2)
test1 = test1.astype(np.float32)
test =test1
```
先算numpy均值，方差与pytorch（注意设置momentum = 1，affine=False）
```python
a1 = 0
v = 0
d = []
std = []
for i in range(3):
    a1 = test1[:,i,:,:]
    d.append(a1.sum())
d = np.array(d)
mean = d/16
m = nn.BatchNorm2d(3,affine=False,momentum=1)
input = torch.from_numpy(test1)
output = m(input)
 
# 均值mean
print('torch 尺寸:',input.shape)
print('pytorch 均值:',m.running_mean.data[0],m.running_mean.data[1],m.running_mean.data[2])
print('手算 均值:',mean)
#numpy 计算： np.mean(np.mean(np.mean(test1,axis=0),axis=1),axis=1)
 
# 方差var = (x-mean) / n
for i in range(3):
    v = test1[:,i,:,:]-mean[i]
    v = ((v**2).sum()/16)**0.5
    std.append(v)
std = np.array(std)
# numpy计算 ：np.std(test1[1,2,3])
print('标准差：',std)
 
#bathnorm
for i in range(3):
    test[:,i,:,:] = (test1[:,i,:,:]-mean[i])/(std[i]+1e-5)
 
print(test[1])
print(output[1])

```
输出：
```python
torch 尺寸: torch.Size([4, 3, 2, 2])
pytorch 均值: tensor(0.4218) tensor(0.5399) tensor(0.3418)
手算 均值: [0.42182693 0.5398935  0.34179664]
标准差： [0.26444062 0.2462885  0.22490181]
```
```python
numpy结果: 
 [[[-1.0535254  -0.4905532 ]
  [-1.3194345  -0.22114275]]

 [[ 0.5717635   1.1570975 ]
  [-1.1665905  -1.5158345 ]]

 [[ 1.6828372   0.8369611 ]
  [ 0.32095668 -0.89949685]]]
pytorch结果：
 tensor([[[-1.0535, -0.4905],
         [-1.3194, -0.2211]],

        [[ 0.5717,  1.1570],
         [-1.1665, -1.5158]],

        [[ 1.6827,  0.8369],
         [ 0.3209, -0.8994]]])
```

可以看出bn的计算是一样的，接下来看一下tf版本的，在这里我加上了**tf.nn.batch_normalitization**
需要将test维度转成**(N, H, W, C)**
```python
x = test1
b = test
x = np.transpose(x,(0,2,3,1))
b = np.transpose(b,(0,2,3,1))
axis = list(range(len(x)-1))
x = tf.convert_to_tensor(x)
wb_mean, wb_var = tf.nn.moments(x,axis)
scale = tf.Variable(tf.ones([3]))
offset = tf.Variable(tf.zeros([3]))
variance_epsilon = 1e-5
Wx_plus_b = tf.nn.batch_normalization(x, wb_mean, wb_var, offset, scale, variance_epsilon)

Wx_plus_b1 = (x - wb_mean) / tf.sqrt(wb_var + variance_epsilon)
Wx_plus_b1 = Wx_plus_b1 * scale + offset

Wx_plus_b2 = tf.layers.batch_normalization(x,momentum=1,scale=False,epsilon= 1e-5)
```
>?	最后对比各种计算结果
```python
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('nn.bn: \n',sess.run(Wx_plus_b[1]))
    print('手算bn：\n',sess.run(Wx_plus_b1[1]))
    print('layers.bn: \n',sess.run(Wx_plus_b2[1]))
    print('numpy bn: \n',b[1])
    print('output: \n',output[1])
```
输出:
```python
nn.bn: 
 [[[-1.0535603   0.57178396  1.6829035 ]
  [-0.49056947  1.1571388   0.83699405]]

 [[-1.319478   -1.1666319   0.32096925]
  [-0.22115016 -1.5158883  -0.8995324 ]]]
手算bn：
 [[[-1.0535601   0.57178396  1.6829034 ]
  [-0.49056944  1.1571388   0.836994  ]]

 [[-1.319478   -1.1666319   0.32096922]
  [-0.22115014 -1.5158883  -0.8995324 ]]]
layers.bn: 
 [[[-1.0535202   0.57176065  1.6828288 ]
  [-0.49055076  1.1570916   0.8369569 ]]

 [[-1.319428   -1.1665846   0.32095507]
  [-0.22114165 -1.5158268  -0.8994923 ]]]
numpy bn: 
 [[[-1.0535254   0.5717635   1.6828372 ]
  [-0.4905532   1.1570975   0.8369611 ]]

 [[-1.3194345  -1.1665905   0.32095668]
  [-0.22114275 -1.5158345  -0.89949685]]]
output: 
 tensor([[[-1.0535, -0.4905],
         [-1.3194, -0.2211]],

        [[ 0.5717,  1.1570],
         [-1.1665, -1.5158]],

        [[ 1.6827,  0.8369],
         [ 0.3209, -0.8994]]])
```

## 结语
对于batchnorm，还是有很多地方不懂或者不理解的。
文章写得比较乱，也并不严谨，有需要改正的也请指出