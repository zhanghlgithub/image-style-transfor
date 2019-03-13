## 介绍 
该项目实现了图像风格转换的功能，vgg16作为特征提取网络，vgg16.npy下载链接:https://pan.baidu.com/s/1gg9jLw3  密码:umce，下载后替换当前的vgg16.npy文件
## 原理
### 内容特征：
每一层卷积作为内容表达，通过训练不断缩小输入图像与某一层feature map的距离，认为学到了该图像的内容表达
### 风格特征：
每一层卷积各channel之间的相似度，每个channel看做一维向量，余弦距离衡量两向量的相似度，两两做内积，输出的结果作为风格特征表达，不断缩小与风格表达之间的距离，认为学到了该图像的风格表达

## 实验结果：
1、通过实验发现，内容特征的表达，在浅层卷积层表现的比较好，风格表达在深层网络表现的更好，训练过程取浅层网络为内容表达，深层网络为风格表达  
2、通过修改content_features，result_content_features，style_features，result_style_features确定哪一层为内容表达和特征表达，也可选择多个层卷积为内容表达和特征表达

## 效果展示：
#### 结果图
![](picture/result.jpg)[](picture/result.jpg)
#### 风格图
![](picture/xingkong.jpg)[](picture/xingkong.jpg)
#### 内容图
![](picture/pengyuyan.jpg)[](picture/pengyuyan.jpg)
