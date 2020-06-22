---
title: 双目摄像头标定与测距(一) 齐次坐标,世界坐标到像素坐标
date: 2019-07-30 15:10:12
tags: '标定原理'
categories: 'slam'
keywords: '坐标变换'
description: '简单介绍摄像头坐标的关系'
top_img: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-382f7d3ffa1be3dd.png
cover: https://cdn.jsdelivr.net/gh/angryhen/picgo_blog_img/blog/15147802-382f7d3ffa1be3dd.png
---

&emsp;&emsp;最近重新研究了一下双目的标定，然而回头一看，不找找代码自己都没法实现出来了，即使有学过但都记忆模糊。果然对新手而言，还是得做笔记。
对于标定而言，如果只是想单纯得到相机参数，用matlab会比opencv实现的效果更准确，matlab自带的相机标定工具箱也是简单上手的，使用教程也有很多资源。以下仅是对原理的简单描述。

## 齐次坐标
&emsp;&emsp;在之前得先了解一下 ‘齐次坐标’ 这个概念，其主要应用在计算机图形计算上，简单理解就是为了简化几何变换在计算机上的计算。例如：

>平移$t$：坐标$(x, y)$分别向上，向右平移$dx,dy$单位，即$(x' - y')=(x+dx,y+dy)$，用矩阵表示：
>
>$$ \left[ {\begin{array}{*{20}{c}}{x'}\\{y'}\end{array}} \right] = \left[ {\begin{array}{*{20}{c}}x\\y
>\end{array}} \right] + \left[ {\begin{array}{*{20}{c}}{dx}\\{dy}\end{array}} \right]\ $$



$ \left[ {\begin{array}{*{20}{c}}{x'}\\{y'}\end{array}} \right] = \left[ {\begin{array}{*{20}{c}}x\\y
\end{array}} \right] + \left[ {\begin{array}{*{20}{c}}{dx}\\{dy}\end{array}} \right] \ $​ 

$$ \Alpha $$

$$E=mc^2$$

