---
title: tensorflow2.0(2)-自定义Dense层以及训练过程
date: 2019-09-27 15:15:11
tags: '自定义训练'
categories: 'tensorflow'
keywords: 'custom train'
description: 'tensorflow2.x在上以keras去搭建网络，这种封装好的基础/高级api在使用上无疑更便捷，但在学习的过程中也不妨自己去实现一些功能，加深理解。'
top_img: https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592830826690&di=8135f644607383569d08b3fe864b7488&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-96cc5340ae6ee2cfdc57d249ac335734_1200x500.jpg
cover: https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592830826690&di=8135f644607383569d08b3fe864b7488&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-96cc5340ae6ee2cfdc57d249ac335734_1200x500.jpg
---

&emsp;&emsp;之前展示了tensorflow2.0的一个初级的实用例子作为开始，对比1.x版本操作还是有好很多的。接下来也将继续从比较基础的层面去了解tf2.0的各种实现
&emsp;&emsp;tensorflow2.0在上以keras去搭建网络，这种封装好的基础/高级api在使用上无疑更便捷，但在学习的过程中也不妨自己去实现一些功能，加深理解。
以实现最简单的全连接层和训练过程为例，

## Dense层

```python
from tensorflow import keras
# 如果在第一层，则需要加入参数：input_shape
keras.layers.Dense(kernel, activation, input_shape)

#反之，一般这么写
keras.layers.Dense(kernel, activation)
```

>kernel: 这一层神经元的个数
>activation：激活函数，一般取'relu'，'selu'也是不错的

简单搭个网络：

```python
model = keras.Sequential([
    keras.layers.Dense(20, activation='relu',input_shape=[224,]),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
```

我们可以用类去自定义Dense的功能，也是非常简单的

```python
class DenseLayer(keras.layers.Layer):
    def __init__(self, kernel, activation=None, **kwargs):
        self.kernel = kernel
        self.activation = keras.layers.Activation(activation)
        super(DenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=(input_shape[1], self.kernel),
                                 initializer='uniform',
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                 shape=(self.kernel,),
                                 initializer='zero',
                                 trainable=True)
        super(DenseLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return self.activation(x @ self.w + self.bias)
```

这一样来，就可以直接用自定义的类DenseLayer去替换keras的全连接层

```python
model = keras.Sequential([
    DenseLayer(20, activation='relu',input_shape=[224,]),
    DenseLayer(10, activation='relu'),
    DenseLayer(2, activation='softmax')
])
```

## 训练过程

### 自定义损失函数

对于实现分类的损失函数而言，也是简单粗暴的，对于标签的格式是one_hot的，用*tf.nn.softmax_cross_entropy_with_logits*，
反之*tf.nn.sparse_softmax_cross_entropy_with_logits*，本文自然用到了后者。

```python
# 传入的logits就是训练数据通过model前向运算一遍得到的结果（model(x)）
def loss_func(logits, label):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=label)
    return tf.reduce_mean(losses)
```

### 自定义梯度更新

关于tf2.0，貌似*tf.GradientTape()*保留了下来，自定义梯度计算这一部分可以作为一个篇章去讲述，以后也会去探索
所以把单步训练和梯度更新过程写在一起

```python
def train_per_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        logit = model(x)
        loss = loss_func(logit, y)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads,
                                  model.trainable_variables))
    return loss
```

### 搭建模型

因为在loss_func的计算里包含了softmax，所以在最后一层不添加激活函数

```python
    model = keras.models.Sequential([
        keras.layers.Reshape(target_shape=(28 * 28, ),
                             input_shape=(28, 28)),
        DenseLayer(200, activation='relu'),
        DenseLayer(300, activation='relu'),
        DenseLayer(100, activation='relu'),
        DenseLayer(10)
    ])
```

### 数据读取

参考上一篇文章，但也有不一样的地方，其中没用到测试集，只关注训练时loss的变化过程

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_train_scaled = tf.cast(x_train_scaled, tf.float32)
y_train = tf.cast(y_train, tf.int32)
train_data = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_train))
train_data = train_data.take(10000).shuffle(10000).batch(256)
```

### 训练

```python
optimizer = keras.optimizers.Adam(lr=0.0003)
epoch = 10
for i in range(epoch):
    for _, (x, y) in enumerate(train_data):
        loss = train_per_step(model, x, y, optimizer)
        print(loss.numpy())
```

最终可以看到loss是降得很快的

end.