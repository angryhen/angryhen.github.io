---
title: tensorflow2.0(3)-Resnet模型
date: 2019-09-30 14:50:23
tags: 
  - resnet
  - tensorflow
categories: 'tensorflow'
keywords: 'tensorflow, resnet'
description: '个人更倾向在实战中学习深化基础，而不是把基础理论学好了再去实践。本篇基于tf2.0是搭建Resnet网络，Resnet有很多变种，也作为很多模型的骨干网络，这次实战项目就从它开始'
top_img: https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592830826690&di=8135f644607383569d08b3fe864b7488&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-96cc5340ae6ee2cfdc57d249ac335734_1200x500.jpg
cover: https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592830826690&di=8135f644607383569d08b3fe864b7488&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-96cc5340ae6ee2cfdc57d249ac335734_1200x500.jpg
---

&emsp;&emsp;tensorflow2不再需要静态建图启动session()，抛弃很多繁杂的功能设计，代码上更加简洁清晰，而在工程上也更加灵活。
但是一些基础的用法，单靠api接口去训练模型是远远无法满足实际的应用，基于这种框架，更多还需要自己在其上自定义开发。

>例如：*model.fit()* 虽然能一句代码把训练跑起来，但你根本无法知道整个模型内部数据的变化，也难以去查看某些变量。我们不可能永远停留在MNIST之类的数据集上。

### Resnet

&emsp;&emsp;个人更倾向在实战中学习深化基础，而不是把基础理论学好了再去实践。本篇基于tf2.x是搭建Resnet网络，Resnet有很多变种，也作为很多模型的骨干网络，这次实战项目就从它开始
（*需要对Resnet有一定的认知了解，本文只是代码实现*）

### 网络结构

&emsp;&emsp; 官方给出的Resnet网络结构，分别为18，34，50，101，152层，可以看出，不同层数之间总体的结构是一样的，这样就很方便用类去实例化每一个模块了
![image]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-dbba113ab30bc1b7.jpg)



### 基础模块

&emsp;&emsp;从conv2_x到conv5_x，18和34layer的结构是一样的，50，101和152是一样的，具体分别为：

![image]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-3d8c1144d9a31096.jpg)
先定义18 or 34 layer的模块

```python
# for 18 or 34 layers
class Basic_Block(keras.Model):

    def __init__(self, filters, downsample=False, stride=1):
        self.expasion = 1
        super(Basic_Block, self).__init__()

        self.downsample = downsample

        self.conv2a = keras.layers.Conv2D(filters=filters,
                                          kernel_size=3,
                                          strides=stride,
                                          kernel_initializer='he_normal',
                                          )
        self.bn2a = keras.layers.BatchNormalization(axis=-1)

        self.conv2b = keras.layers.Conv2D(filters=filters,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer='he_normal'
                                          )
        self.bn2b = keras.layers.BatchNormalization(axis=-1)

        self.relu = keras.layers.ReLU()

        if self.downsample:
            self.conv_shortcut = keras.layers.Conv2D(filters=filters,
                                                     kernel_size=1,
                                                     strides=stride,
                                                     kernel_initializer='he_normal',
                                                     )
            self.bn_shortcut = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = self.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu(x)

        if self.downsample:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs

        x = keras.layers.add([x, shortcut])
        x = self.relu(x)
```

代码虽然长了点，但看一下call() 里面就很清晰了，就是2个 *conv+bn+relu*，最后与input做点加操作
同理应用在50，101 or 152 layer：

```python
# for 50, 101 or 152 layers
class Block(keras.Model):

    def __init__(self, filters, block_name,
                 downsample=False, stride=1, **kwargs):
        self.expasion = 4
        super(Block, self).__init__(**kwargs)

        conv_name = 'res' + block_name + '_branch'
        bn_name = 'bn' + block_name + '_branch'
        self.downsample = downsample

        self.conv2a = keras.layers.Conv2D(filters=filters,
                                          kernel_size=1,
                                          strides=stride,
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2a')
        self.bn2a = keras.layers.BatchNormalization(axis=3, name=bn_name + '2a')

        self.conv2b = keras.layers.Conv2D(filters=filters,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2b')
        self.bn2b = keras.layers.BatchNormalization(axis=3, name=bn_name + '2b')

        self.conv2c = keras.layers.Conv2D(filters=4 * filters,
                                          kernel_size=1,
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2c')
        self.bn2c = keras.layers.BatchNormalization(axis=3, name=bn_name + '2c')

        if self.downsample:
            self.conv_shortcut = keras.layers.Conv2D(filters=4 * filters,
                                                     kernel_size=1,
                                                     strides=stride,
                                                     kernel_initializer='he_normal',
                                                     name=conv_name + '1')
            self.bn_shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name + '1')

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x)

        if self.downsample:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs

        x = keras.layers.add([x, shortcut])
        x = tf.nn.relu(x)

        return x
```

>对于downsample的操作，如果input和最后一层输出的chanels不一样就需要downsample来保持chanel一致，这样才能相加，一般解析resnet的文章都会提到。
>用类封装了模块的功能，接下来只需要在主体网路结构里添加这个模块就好了

### 主体结构

用subclassing的方式去搭建model，就像砌墙一样，一个模块一个模块拼上去就好了，先在__init__()里面定义好需要用到的方法，再在call()把他们调用起来。
对于resnet的主体结构，先看一下call()里是该如何写的：

```python
def call(self, inputs, **kwargs):
    x = self.padding(inputs)
    x = self.conv1(x)
    x = self.bn_conv1(x)
    x = tf.nn.relu(x)
    x = self.max_pool(x)

    # layer2
    x = self.res2(x)
    # layer3
    x = self.res3(x)
    # layer4
    x = self.res4(x)
    # layer5
    x = self.res5(x)

    x = self.avgpool(x)
    x = self.fc(x)
    return x
```

>一目了然，跟文章开头的结构图一摸一样，
>最重要的是中间*conv2-5* 的操作，这个需要对resnet结构熟悉

在Resnet的__init__()里面，这样去定义中间的4个层

```python
# layer2
self.res2 = self.mid_layer(block, 64, layers[0], stride=1, layer_number=2)

# layer3
self.res3 = self.mid_layer(block, 128, layers[1], stride=2, layer_number=3)

# layer4
self.res4 = self.mid_layer(block, 256, layers[2], stride=2, layer_number=4)

# layer5
self.res5 = self.mid_layer(block, 512, layers[3], stride=2, layer_number=5)
```

函数*self.mid_layer()* 就是把block模块串起来

```python
def mid_layer(self, block, filter, block_layers, stride=1, layer_number=1):
    layer = keras.Sequential()
    if stride != 1 or filter * 4 != 64:
        layer.add(block(filters=filter,
                        downsample=True, stride=stride,
                        block_name='{}a'.format(layer_number)))

    for i in range(1, block_layers):
        p = chr(i + ord('a'))
        layer.add(block(filters=filter,
                        block_name='{}'.format(layer_number) + p))

    return layer
```

到此主体的结构就定义好了，官方源码Resnet，是直接从上到下直接编写的，就是一边构建网络一边计算,类似于这样

```python
x = input()
x = keras.layers.Conv2D()(x)
x = keras.layers.MaxPooling2D()(X)
x = keras.layers.Dense(num_classes)(x)
```

&emsp;&emsp;相对来说更喜欢用subclassing的方式去搭建model，虽然代码量多了点，但是结构清晰，自己要中间修改的时候也很简单，也方便别的地方直接调用，但有一点不好就是，当想打印模型*model.summary()* 的时候，看不到图像在各个操作后的shape，直接显示multiple，目前不知道有没其他的方法。。
![image]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-97f82296ad370733.png)

### 代码

&emsp;&emsp;上述代码呈现了Resnet的大部分内容，可以随便实现18-152layer，全部代码放在了我的github里：https://github.com/angryhen/learning_tensorflow2.0/blob/master/base_model/ResNet.py

&emsp;&emsp;持续更新中，tensorflow2.x这一系列的代码也会放在上面，包括VGG，Mobilenet的基础网络，以后也会更新引入senet这种变种网络。

Thanks