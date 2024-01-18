# DCGAN-and-WGAN
Implementing and Improving DCGAN and WGAN with PyTorch


## 实验环境

本地部署：

```
conda create –n env_name python=3.8 # 创建虚拟环境，env_name为环境名称

conda activate env_name # 激活新建的环境

conda install pytorch torchvision # 安装pytorch 和 torchvision
```

云端部署：

<img src="C:\Users\李心怡\AppData\Roaming\Typora\typora-user-images\image-20240118101157511.png" alt="image-20240118101157511" style="zoom:50%; float: left;" />

## 数据集下载

1. MNIST数据集：使用torchvision.datasets直接加载

2. 二次元动漫头像数据集

   下载链接：[Anime Faces (kaggle.com)](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)

## 运行方式

文件目录结构示例如下：
├─128-1                                # batch_size=128 learning_rate=0.0001
│  ├─128-1.ipynb                # 可以直接运行的jupyter文件，含有loss曲线图和置信度曲线图
│  ├─output.txt                   # 迭代过程的loss和D(x)结果
│  └─pic                                # 迭代每一轮次生成的图片                   

所有.ipynb文件均可在本地jupyter notebook和云端笔记本直接打开运行

## 实验结果

A. MNIST手写数据集

(1) 迭代次数epoch对图片生成质量的影响

|                           epoch=1                            |                           epoch=40                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps57.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps58.jpg) |

|                           Loss曲线                           |                          置信度曲线                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps4.jpg) | ![](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps5.jpg) |

(2) 批次大小batchsize对图片生成质量的影响

lr为0.002，epoch为30时，对比batchsize分别为128,512,1024时的图片生成质量：

|                             128                              |                             512                              |                             1024                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps6.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps9.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps12.jpg) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps8.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps10.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps13.jpg) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps7.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps11.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps14.jpg) |

(3) 学习率lr对对图片生成质量的影响

batchsize=128，epoch=30时对比学习率为0.0001,0.0002,0.0003时的图片质量：

|                            0.0001                            |                            0.0002                            |                            0.0003                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20240118105400430](C:\Users\李心怡\AppData\Roaming\Typora\typora-user-images\image-20240118105400430.png) | ![image-20240118105339132](C:\Users\李心怡\AppData\Roaming\Typora\typora-user-images\image-20240118105339132.png) | ![image-20240118105422580](C:\Users\李心怡\AppData\Roaming\Typora\typora-user-images\image-20240118105422580.png) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps16.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps20.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps22.jpg) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps17.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps19.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps23.jpg) |

B. 二次元动漫头像数据集

(1) 迭代次数epoch对图片生成质量的影响

|                           epoch=10                           |                           epoch=50                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps24.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps25.jpg) |
|                          epoch=100                           |                          epoch=200                           |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps26.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps27.jpg) |

迭代过程中生成器和判别器的Loss曲线和置信度曲线如下所示：

|                             Loss                             |                            置信度                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps28.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps29.jpg) |

(2) 批次大小batchsize对图片生成质量的影响

当学习率lr为0.002，epoch为30时，对比batchsize分别为128,512,1024时的图片生成质量：

|                             128                              |                             512                              |                             1024                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps59.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps60.jpg) | ![image-20240118105936720](C:\Users\李心怡\AppData\Roaming\Typora\typora-user-images\image-20240118105936720.png) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps62.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps63.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps64.jpg) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps65.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps66.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps67.jpg) |

(3) 学习率lr对对图片生成质量的影响

batchsize=128，epoch=30时对比学习率为0.0001,0.0002,0.0003时的图片质量：

|                            0.0001                            |                            0.0002                            |                            0.0003                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps68.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps69.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps70.jpg) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps71.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps72.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps73.jpg) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps74.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps75.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps76.jpg) |

(4) 改进优化器的影响

原模型使用的是Adam优化器，尝试将优化器改为RMSprop后，在batchsize=128，lr=0.0002的参数下迭代30轮次，生成的图片如下所示：

![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps77.jpg) 

迭代过程中生成器和判别器的Loss曲线和置信度曲线如下所示：

![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps78.jpg)![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps79.jpg)

3.2 WGAN

分别使用WGAN和改进后的WGAN-GP和WGAN-DIV，在epoch=30，batchsize=64，lr=0.00005参数设置下进行图像生成实验：

| WGAN                                                         | WGAN-GP                                                      | WGAN-DIV                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps51.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps52.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps53.jpg) |
| ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps54.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps55.jpg) | ![img](file:///C:\Users\李心怡\AppData\Local\Temp\ksohtml2916\wps56.jpg) |
