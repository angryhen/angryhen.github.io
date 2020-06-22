---
title: tensorflow2.0牛刀小试--LetNet+fashion_Mnist
date: 2019-01-02 16:10:19
tags: '分类'
categories: 'tensorflow'
keywords: '深度学习'
description: '这是学习tensorflow2.x 的系列开始'
top_img: https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592830826690&di=8135f644607383569d08b3fe864b7488&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-96cc5340ae6ee2cfdc57d249ac335734_1200x500.jpg
cover: https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592830826690&di=8135f644607383569d08b3fe864b7488&imgtype=0&src=http%3A%2F%2Fpic4.zhimg.com%2Fv2-96cc5340ae6ee2cfdc57d249ac335734_1200x500.jpg
---



&emsp;&emsp;回想以前刚接触tensorflow，在各种API以及各种花里胡哨用法的打击下，对于小白的我果断放弃转而投奔pytorch的怀抱。几个框架学习过程中也没做笔记，tensorflow2.x的出世（出了一年了吧。），跟pytorch一样动态图的机制，看来有机会*import tensorflow as torch*了。借此机会回来重新学习tensorflow，实现一些简单的操作，温故知新。
&emsp;&emsp;在tf2.x里面，默认开启了eager Execution，并且使用Keras作为默认高级API，网络的搭建和自定义操作也变得更简单，估计到处飞的keras会引来吐槽吧。

## 加载Mnist

>keras.datasets ：内置了7个基本的数据集，分别是
> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; boston_housing，cifar10，cifar100，fashion_mnist，imdb，mnist，reuters
>对于新手有一个学习的地方就是关于数据的基本读取方法，每个数据集都内置了load_data()方法（包括了下载与加载）
>例如fashion_mnist:
```python

@keras_export('keras.datasets.fashion_mnist.load_data')
def load_data():
  """Loads the Fashion-MNIST dataset.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  License:
      The copyright for Fashion-MNIST is held by Zalando SE.
      Fashion-MNIST is licensed under the [MIT license](
      https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

  """
  dirname = os.path.join('datasets', 'fashion-mnist')
  base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)

```
所以我们加载的时候只要：
```python
(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_val, x_train = x_train_all[:3000], x_train_all[3000:]
y_val, y_train = y_train_all[:3000], y_train_all[3000:]
```
## 归一化
&emsp;&emsp;数据加载完后，对数据归一化，而归一化的作用在此就不展开阐述了。在此利用scikit-learn库进行处理：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
                    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_val_scaled = scaler.transform(
                    x_val.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_test_scaled = scaler.transform(
                    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1）
```
这一步操作展开解释：
1）`StandardScaler`是用来计算归一化和标准差的类，code里为何训练集用了`fit_transform`，而测试集则用了`transform`，从源码的注释里可以看到：
>    Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the `transform` method.
>    ![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-9c9a73218e43095b.png)
>     储存在训练集计算得来的平均值和标准差 ，以便在后面使用`transform`方法（意思就是在`transform`时利用了`fit_transform`的平均值和标准差，还因为数据的分布都是相同的，这也得到了解释）

2） `reshape(-1, 1)` → `transform` → `reshape(-1, 28, 28, 1)`
为何不直接做`transform`而是`reshape(-1, 1)，看下面例子就知道了：

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 8]])
b = a.reshape(-1, 1)
print(scaler.fit_transform(a))
print(scaler.fit_transform(b))

>>[[-1.22474487 -1.22474487 -1.29777137]
   [ 0.          0.          0.16222142]
   [ 1.22474487  1.22474487  1.13554995]]

>>[[-1.60422237]
   [-1.19170805]
   [-0.77919372]
   [-0.3666794 ]
   [ 0.04583492]
   [ 0.45834925]
   [ 0.87086357]    
   [ 1.2833779 ]
   [ 1.2833779 ]]
```
需要对全局的数据进行处理，而不是各个维度上分别处理。最后`reshape(-1, 28, 28, 1)`是为了后面输入到网络中去计算。
## 网络结构
&emsp;&emsp;网络的搭建用了Sequential，官方文档定义为序贯模型，也就是最简单，线性堆叠，例如VGG这样一路走到黑的模型。对于更复杂的模型，支持多输入多输出，层与层之间想怎么连怎么连，就要用到Model。显然，Sequential只是作为特殊的一类单独拿出来用了。

下面参照了LetNet-5构建了模型，只是输入改成了28*28
```python
# model
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=6, kernel_size=5, strides=1,
                        activation='relu',
                        input_shape=[28, 28, 1]),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=16, kernel_size=5, strides=1,
                        activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=120, kernel_size=3, strides=1,
                        activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(84, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```
### 模型训练的BP模式设置

```
adam = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
```
&emsp;&emsp;对于loss的选择
  如果标签为0，1，2，3···的数字编码，则用`sparse_categorical_crossentropy`,
如果为[0, 0, 1, 0, 0]这类的独热编码，则用`categorical_crossentropy`
####训练以及参数设置
使用`fit`函数进行参数的设置以及训练，都是比较基本的使用方法，相对都是通用的
```python
history = model.fit(x_train_scaled, y_train, epochs=100,
                    validation_data=[x_val_scaled, y_val],
                    batch_size=256)
```
### 可视化

```python
def show_result(history):
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
show_result(history)
```
![result.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-e4604fbdb5d00d04.png)
感觉过拟合了。。训练太久，不过单从数据也不能说明什么
下面试着用tensorflow的`from_tensor_slices`读取数据，训练获得更好的效果

```
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_train))
train_dataset = train_dataset.repeat(100).batch(256).prefetch(100)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val_scaled, y_val))
val_dataset = val_dataset.repeat(100).batch(256)

# train
history3 = model.fit(train_dataset, epochs=100,
                    validation_data=train_dataset,
                    validation_steps=len(x_val) // 256,
                    steps_per_epoch= len(x_train) // 256
                    )
```
![C7`IOVMBLT9NFWX1@P6F.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog2/15147802-3f9e9b5beace4324.png)

得到了更好的一个结果，其中的原理网上很多资源，太懒写不动
>有一点感觉就是：学习的路上总会遇到很多坑，不必要太纠结要搞懂里面的所有原理，有些理论看不懂但我们可以先动手，回头看会往往发现会有新的体会和理解，切勿死磕。就像学英语一样，一直学语法记单词并不能帮助我们更好的说出来，反而经常去说更能辅助我们学深层的语法（现在的小孩学英语很多都不先学音标了，而是像拼音一样的拼读法，先培养语感真的很重要）

更多有关tf2.0的东西以后慢慢更，仅以此纪录自己的学习过程，也欢迎来跟我讨论O(∩_∩)O