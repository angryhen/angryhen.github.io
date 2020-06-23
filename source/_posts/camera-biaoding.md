---
title: 双目摄像头标定与测距(一) 齐次坐标,世界坐标到像素坐标
date: 2019-07-30 15:10:12
tags: '标定原理'
categories: 'slam'
keywords: '坐标变换'
description: '简单介绍摄像头坐标的关系'
top_img: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-6699055c965e22eb.png
cover: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-6699055c965e22eb.png
---

&emsp;&emsp;最近重新研究了一下双目的标定，然而回头一看，不找找代码自己都没法实现出来了，即使有学过但都记忆模糊。果然对新手而言，还是得做笔记。
对于标定而言，如果只是想单纯得到相机参数，用matlab会比opencv实现的效果更准确，matlab自带的相机标定工具箱也是简单上手的，使用教程也有很多资源。以下仅是对原理的简单描述。

## 齐次坐标

&emsp;&emsp;在之前得先了解一下 ‘齐次坐标’ 这个概念，其主要应用在计算机图形计算上，简单理解就是为了简化几何变换在计算机上的计算。例如：

>平移$t$：坐标$(x, y)$分别向上，向右平移$dx,dy$单位，即$(x' - y')=(x+dx,y+dy)$，用矩阵表示：
>
>$$\left[\left.\begin{array}{c}x'\\y'\end{array}\right]=\begin{bmatrix}x\\y\end{bmatrix}+\left[\begin{array}{c}dx\\dy\end{array}\right.\right]$$
>
>旋转$R$：在二维空间中，对于旋转角度为$\theta$，有
>
>$$\begin{bmatrix}x'\\y'\end{bmatrix}=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}\ast\begin{bmatrix}x\\y\end{bmatrix}$$
>
>为了合并矩阵运算的乘法与加法，所以引用了**齐次坐标**，即：
>$$\begin{bmatrix}x'\\y'\\1\end{bmatrix}\;=\;\begin{bmatrix}\cos\theta&-\sin\theta&0\\\sin\theta&\cos\theta&0\\0&0&1\end{bmatrix}\;\ast\;\begin{bmatrix}1&0&tx\\0&1&ty\\0&0&1\end{bmatrix}\;\ast\;\begin{bmatrix}x\\y\\1\end{bmatrix}\;=\;\begin{bmatrix}\cos\theta&-\sin\theta&(1-\cos\theta)tx+ty\ast\sin\theta\\\sin\theta&\cos\theta&(1-\cos\theta)ty-tx\ast\sin\theta\\0&0&1\end{bmatrix}\;\ast\;\begin{bmatrix}x\\y\\1\end{bmatrix}$$
>另一方面，在实际的投影空间里笛卡尔坐标难以进行表达，因为两条平行线会绝对平行，放在投影空间则不然，例如笛卡尔坐标下笔直平行的马路在视野里最终会无限逼近于无穷远一点，楼再建高点就能互通了。。

| ![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-4a97b0d4c8710749.png) ![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-6699055c965e22eb.png)

&emsp;&emsp;齐次坐标就是讲原来n维的向量用n+1维表示，用$p'=Ap$($A$指$R$和$t$组成的转换矩阵)方式表示几何变换。将$(x, y)$附加第三个坐标，于是每个坐标用一个三元组$(x,y,w)$表示，称为点$(x,y)$的齐次坐标。一般来说当$w$不为0时，采用$w=1$，并将$(\frac{x}{w},\frac{y}{w})$称为齐次点$(x,y,w)$的笛卡尔坐标。
*有关齐次坐标这里只是片面的解释与理解*   

## 坐标转换关系图

![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-dc671381bd3582e4.png)

### 世界坐标与相机坐标

![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-1a7f891206bfc2f6.png)

&emsp;从世界坐标系$(X_w,\;Y_w,\;Z_w)$转换到相机坐标系，属于刚体变换(在三维空间中， 把一个几何物体作旋转，平移的运动，称之刚体变换)，参考齐次坐标的示范，当拓展到三维时 $\left(X_W,Y_W,Y_W\right)\rightarrow\left(X_C,Y_C,Z_C\right)$

$$\begin{bmatrix}X_C\\Y_C\\Z_C\\1\end{bmatrix}\;=\;\begin{bmatrix}l_{00}&l_{01}&l_{02}&tx\\l_{10}&l_{11}&l_{12}&ty\\l_{20}&l_{21}&l_{22}&tz\\0&0&0&0\end{bmatrix}\;\ast\;\begin{bmatrix}X_W\\Y_W\\Z_W\\1\end{bmatrix}\;=\;\begin{bmatrix}R&T\\0&1\end{bmatrix}\;\ast\;\begin{bmatrix}X_W\\Y_W\\Z_W\\1\end{bmatrix}$$

为3x3的旋转矩阵，$T$为平移向量，$R,T$两个外参组成世界坐标到相机坐标的转换矩阵

### 相机坐标到图像坐标

![image.png]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-bf54f8f1b11f2bff.png)

&emsp;&emsp;摄影机坐标系的原点为摄像机光心，x轴与y轴与图像的X,Y轴平行，z轴为摄像机光轴，它与图像平面垂直，以此构成的空间直角坐标系称为摄像机坐标系，也称为相机坐标系，摄像机坐标系是三维坐标系。
&emsp;&emsp;从相机坐标系到图像坐标系，是3D到2D的转换，属于**透视投影**，下图更清晰表现出来![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-382f7d3ffa1be3dd.png)

> 由相似三角形，得:
>
> $$  \frac{AB}{oC}\;=\;\frac{PB}{pC}\;=\;\frac{AO_C}{oO_C}\rightarrow\frac{X_C}x\;=\;\frac{Yc}y=\frac{Z_C}f\\\;\;\;\;\;\;\;\;\;\;\;\;\;\;\downarrow\\x\;=\;f\frac{X_C}{Z_C},\;\;\;\;y\;=\;f\frac{Y_C}{Z_C} $$
>
> 通过变换可得：S
>
> $$Z_C\begin{bmatrix}x\\y\\1\end{bmatrix}\;=\;\begin{bmatrix}xZ_C\\yZ_C\\Z_C\end{bmatrix}\;=\;\begin{bmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{bmatrix}\;\ast\;\begin{bmatrix}X_C\\Y_C\\Z_C\\1\end{bmatrix}$$
>
> 鉴于文章开头齐次坐标的概念，$Z_c$作为系数，在图像坐标系上，$\begin{bmatrix}x\\y\\1\end{bmatrix}\;=\;Z_C\begin{bmatrix}x\\y\\1\end{bmatrix}\;$

### 图像坐标到像素坐标

&emsp;&emsp;在做图像处理的时候，我们对于图片像素点的读取，都是以图像左上角为原点，向右向下分别为$x,y$ 的正方向，为了方便操作，我们需要从图像坐标转换到像素坐标。 ![]( https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-6cff963e1ca63e11.png)

其中${u_0},{v_0}$代表图像坐标原点在像素坐标的位置，图像坐标的单位是mm，像素坐标的单位是pix，对于两者的转换，${d_x},{d_y}$分别表示一个pix在行列上等于多少个mm。对此则有：

$$\begin{Bmatrix}u=\frac x{dx}+u_0\\v=\frac y{dy}+v_0\end{Bmatrix}\;\begin{bmatrix}u\\v\\1\end{bmatrix}\;=\;\begin{bmatrix}\frac1{dx}&0&u_0\\0&\frac1{dy}&v_0\\0&0&1\end{bmatrix}\;\ast\;\begin{bmatrix}x\\y\\1\end{bmatrix}$$

## 总结

&emsp;&emsp;如此一来，相机的成像原理以及对应的坐标关系可以梳理成：

$$Z_C\begin{bmatrix}u\\v\\1\end{bmatrix}\;=\;\begin{bmatrix}\frac1{dx}&0&u_0\\0&\frac1{dy}&v_0\\0&0&1\end{bmatrix}\;\ast\;Z_C\begin{bmatrix}x\\y\\1\end{bmatrix}\\=\begin{bmatrix}\frac1{dx}&0&u_0\\0&\frac1{dy}&v_0\\0&0&1\end{bmatrix}\;\begin{bmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{bmatrix}\;\ast\;\begin{bmatrix}X_C\\Y_C\\Z_C\\1\end{bmatrix}\\=\begin{bmatrix}\frac1{dx}&0&u_0\\0&\frac1{dy}&v_0\\0&0&1\end{bmatrix}\;\begin{bmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{bmatrix}\;\begin{bmatrix}l_{00}&l_{01}&l_{02}&t_x\\l_{10}&l_{11}&l_{12}&t_y\\l_{20}&l_{21}&l_{22}&t_z\\0&0&0&1\end{bmatrix}\;\;\ast\;\;\begin{bmatrix}X_W\\Y_W\\Z_W\\1\end{bmatrix}$$

> 其中，相机内参：$\begin{bmatrix}\frac1{dx}&0&u_0\\0&\frac1{dy}&v_0\\0&0&1\end{bmatrix}\;\begin{bmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{bmatrix}\;$,
>
> 相机外参: $\;\begin{bmatrix}l_{00}&l_{01}&l_{02}&t_x\\l_{10}&l_{11}&l_{12}&t_y\\l_{20}&l_{21}&l_2&t_z\\0&0&0&1\end{bmatrix}\;$

以上是不考虑畸变情况下的转换，关于标定矫正将会在下一篇继续研究

参考：[https://www.cnblogs.com/zyly/p/9366080.html](https://www.cnblogs.com/zyly/p/9366080.html)