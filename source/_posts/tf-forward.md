---
title: tensorflow2.0(4)-前向传播计算
date: 2019-10-01 05:12:21
tags: 
  - tensorflow
categories: 'tensorflow'
keywords: 'tensorflow'
description: '在网络的前向计算中，我们都可以用y = x@w + b 的形式去描述，此文介绍如何用tensorflow2.0计算网络中的前向计算和参数更新'
top_img: https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592830826690&di=8135f644607383569d08b3fe864b7488&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-96cc5340ae6ee2cfdc57d249ac335734_1200x500.jpg
cover: https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592830826690&di=8135f644607383569d08b3fe864b7488&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-96cc5340ae6ee2cfdc57d249ac335734_1200x500.jpg
---

&emsp;&emsp;在网络的前向计算中，我们都可以用$y = x*w + b$ 的形式去描述，此文介绍如何用tensorflow2.0计算网络中的前向计算和参数更新

### 前向计算

#### tf.constant() and tf.Variable()

&emsp;&emsp;tf2中有两种创建张量的方式，分别为**tf.constant()**和**tf.Variable()**

>**tf.constant()**: 是创建一个常量数值/列表tensor，当然创建后值是不可变的，一般定义网络中不需要更新的参数。
>**tf.Variable()**:  类型在普通的张量类型基础上添加了 *name*，*trainable* 等属性来支持计算图的构建，对于需要计算梯度并优化的张量，要通过此函数封装

简单的试验即可窥探两者的关系

```python
x = tf.constant(1.)
<tf.Tensor: id=5732, shape=(), dtype=float32, numpy=1.0>

y = tf.Varible(1.)
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
print(y.name, y.trainable)
Variable:0 True
```

在tf.Variable中，trainable参数默认是True，这里我们便可以根据情况手动设置True/False
constant也可以转换到Variable

```python
x = tf.constant(1.)
<tf.Tensor: id=5732, shape=(), dtype=float32, numpy=1.0>

y = tf.Variable(x)
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
```

对于$w$和$b$，用**tf.Variable**定义，数据依然使用fashion_mnist

```python
# 数据准备
x, y = keras.datasets.fashion_mnist.load_data()[0]  
x = tf.reshape(x, [-1, 28 * 28])   # 转换格式
x = tf.cast(x, tf.float32)
x = x / 255.
y = tf.one_hot(y, depth=10)  # 转换成独热编码

# 参数定义
w1 = tf.Variable(tf.random.truncated_normal([784, 128], stddev=0.1))
b1 = tf.Variable(tf.zeros([128]))

w2 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
```

#### tf.GradientTape()

&emsp;&emsp;在使用 TensorFlow2自动求导功能计算梯度时，需要将前向计算过程放置在 tf.GradientTape()环境中，从而利用 GradientTape 对象的 gradient()方法自动求解参数的梯 度，并利用 optimizers 对象更新参数,形式如下

```python
with tf.GradientTape() as tape:
    h1 = x@w1 + b1
    h1 = tf.nn.relu(h1)
    
    out = h1@w2 +b2
    
    loss = tf.square(y - out)
    loss = tf.reduce_mean(loss)

grads = tape.gradient(loss, [w1, b1, w2, b2])
print(grads)

[<tf.Tensor: id=5731, shape=(784, 128), dtype=float32, numpy=
 array([[-1.6100607e-03,  6.9914912e-03,  5.4573239e-04, ...,
          1.1394346e-02,  6.7010368e-03, -4.5570749e-04],
        [-4.6598859e-02,  8.9481764e-02,  1.0636392e-02, ...,
          8.8637583e-02,  6.8306737e-02,  3.1169101e-03],
        [-2.4604234e-01,  3.0288315e-01,  1.2252450e-02, ...,
          5.1393658e-01,  2.8754672e-01, -1.1293454e-02],
        ...,
        [-1.7132786e+01,  2.6959139e+01,  1.4552463e+00, ...,
          3.1937271e+01,  3.0912199e+01, -6.0341411e+00],
        [-5.6701980e+00,  7.6560483e+00,  5.6733060e-01, ...,
          8.5440788e+00,  9.5710907e+00, -1.1315466e+00],
        [-5.4063934e-01,  6.1388338e-01,  9.2204645e-02, ...,
          6.7820299e-01,  8.7314850e-01, -1.0959221e-01]], dtype=float32)>,
 <tf.Tensor: id=5730, shape=(128,), dtype=float32, numpy=
 array([-7.42730665e+00,  1.12210531e+01,  1.39284635e+00,  1.01494350e+01,
        -8.07421327e-01,  1.23828259e+01, -6.74541807e+00,  4.68487740e+00,
         2.48209095e+00,  6.72132134e-01,  1.45874703e+00,  3.87370616e-01,
         4.15165663e+00, -2.76716614e+00, -1.40455708e-01,  2.00560951e+00,
         2.68204361e-01, -9.28761959e+00, -8.48146820e+00,  4.43407488e+00,
         2.67940640e+00,  1.73321190e+01, -9.16018337e-02, -5.43434918e-01,
         2.07012024e+01,  2.75893402e+00,  1.56505895e+00,  2.19261336e+00,
         1.77217662e+00, -6.82915926e+00,  1.27866373e+01, -3.91614413e+00,
        -6.03287840e+00, -7.77332306e-01,  7.70938247e-02,  2.78442907e+00,
        -4.03842735e+00,  1.42863894e+00,  6.28497660e-01, -2.08575636e-01,
         1.29537325e+01,  1.57910907e+00,  6.15310764e+00, -1.28530553e-02,
        -5.76123714e-01,  8.60795438e-01,  1.76193161e+01,  7.69186687e+00,
         1.05096633e-02,  5.15164211e-02, -2.48046803e+00,  1.32270586e+00,
        -2.25072289e+00,  5.69219828e+00, -1.38959253e+00,  1.62736397e+01,
        -8.12963390e+00,  2.23423982e+00,  7.58430099e+00,  2.93936163e-01,
         4.16086674e+00,  9.06507683e+00,  7.49802440e-02,  9.94850695e-01,
        -5.30858874e-01,  2.83726931e+00,  6.86642528e-01,  1.60034144e+00,
         1.67230380e+00,  1.03161788e+00,  1.59616947e+01,  1.31335080e+00,
         4.84881115e+00,  6.14683032e-01, -8.90313816e+00, -2.12549075e-01,
        -2.11784393e-01, -1.48440564e+00,  4.82855380e-01, -9.86536026e-01,
        -3.39097095e+00, -6.08709872e-01,  6.58010149e+00, -1.88684082e+00,
        -1.24421378e-03,  1.05124637e-01,  5.68092060e+00, -5.80791092e+00,
         1.15198154e+01,  5.63493919e+00,  5.89573622e-01,  6.79265213e+00,
         6.17962408e+00, -8.74048519e+00,  5.99117374e+00,  1.38388929e+01,
        -3.47207069e-01,  4.13742256e+00,  9.31134319e+00,  7.82230973e-01,
         1.00510216e+01,  6.51883888e+00,  4.51659933e-02,  2.16893425e+01,
         1.84543401e-01, -3.50218683e-01,  9.49227095e-01,  4.19830494e-02,
        -8.80231977e-01,  1.04302864e+01,  1.07236528e+01, -3.01085401e+00,
         1.77256107e+00, -1.66283143e+00, -8.47181702e+00, -2.34353259e-01,
         1.53497944e+01,  7.91017103e+00,  1.79229784e+00,  5.81019521e-01,
         3.54849339e-01,  1.95794022e+00,  1.26474485e+01, -3.29144746e-01,
         1.15832796e+01,  1.18883247e+01,  8.93653870e+00, -8.44317555e-01],
       dtype=float32)>,
 <tf.Tensor: id=5721, shape=(128, 10), dtype=float32, numpy=
 array([[ 3.1336699e+03,  1.8366328e+03, -2.4875808e+03, ...,
         -4.4612781e+02, -2.6972102e+03,  4.0494846e+03],
        [ 1.2611420e+04,  9.4572920e+03, -8.0339561e+03, ...,
         -3.4947998e+02, -9.3134062e+03,  1.3998470e+04],
        [ 3.3486831e+03,  2.8550195e+03, -2.3804250e+03, ...,
         -3.6743832e+02, -3.3628354e+03,  4.0677268e+03],
        ...,
        [ 7.8448711e+03,  5.0678994e+03, -4.8276855e+03, ...,
         -1.4275117e+03, -7.3473486e+03,  1.0647826e+04],
        [ 1.6595094e+04,  1.1535066e+04, -1.0277844e+04, ...,
         -1.7369330e+03, -1.4152996e+04,  1.9956799e+04],
        [ 1.9223188e+03,  1.3224614e+03, -1.4478511e+03, ...,
         -5.4362202e+00, -1.2794425e+03,  2.1074729e+03]], dtype=float32)>,
 <tf.Tensor: id=5719, shape=(10,), dtype=float32, numpy=
 array([ 52.70592  ,  34.86287  , -35.887447 ,   2.6505039,  39.521217 ,
        -13.3063545,  46.65375  ,  -2.53192  , -39.36912  ,  62.809788 ],
       dtype=float32)>]
```

grads为一个列表, 分别记录了$w1$,$b1$,$w2$,$b2$的梯度

#### 更新参数

熟悉深度学习里面,参数的更新形式如下:$${w}' = w - lr*\frac{\vartheta l}{\vartheta w}$$
用assign_sub更新参数

```python
w1.assign_sub(lr * grads[0]) 
b1.assign_sub(lr * grads[1]) 
w2.assign_sub(lr * grads[2]) 
b2.assign_sub(lr * grads[3])
```

当我们不断的循环更新参数,理论上loss的值是不断减少的

```python
for i in range(50):
    with tf.GradientTape() as tape:
        h1 = x@w1 + b1
        h1 = tf.nn.relu(h1)

        out = h1@w2 +b2

        loss = tf.square(y - out)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, [w1, b1, w2, b2])
    lr = 0.01
    w1.assign_sub(lr * grads[0]) 
    b1.assign_sub(lr * grads[1]) 
    w2.assign_sub(lr * grads[2]) 
    b2.assign_sub(lr * grads[3])
    print(loss.numpy())

loss:
0.41989234
0.35508955
0.31600225
0.29151577
0.27551207
0.26453122
0.25657973
0.25048772
0.24556401
0.24139453
0.23772821
0.2344122
0.2313513
0.22848529
0.22577567
0.2231966
0.22073068
...
```

实验证明也如此
当网络参数多的时候,不肯能再用assign_sub一个个地计算,这时候就要用到**optimizer**了

```python
optimizer = keras.optimizers.SGD()
for i in range(50):
    with tf.GradientTape() as tape:
        h1 = x@w1 + b1
        h1 = tf.nn.relu(h1)

        out = h1@w2 +b2

        loss = tf.square(y - out)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, [w1, b1, w2, b2])
    optimizer.apply_gradients(zip(grads, [w1, b1, w2, b2]))
```

#### end

&emsp;&emsp;此文写的东西比较简单基础,但也是非常重要的部分,操作再骚也离不开如此,熟练理解将很好帮助模型代码的构建