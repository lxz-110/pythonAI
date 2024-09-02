## 1. 效果图

![效果图](https://p.sda1.dev/13/d828a0c0b4cad1a0d942b82612b47869/QQ截图20231010234218.png)

- *注：1.每台电脑的配置及系统不同实际画面及效果可能有出入。*
-    *2.本项目存在随机性，由于神经网络的不确定性导致的预测结果和准确结果可能不同。*

## 2. 项目主题

- **项目名称:** Python Neural networks-手写数字识别
- **设计说明:** 本项目旨在构建一个用于手写数字识别的神经网络应用程序。通过神经网络的训练和优化，用户可以绘制手写数字，系统将预测输入数字的正确标签。这个应用程序不仅可以用于识别手写数字，随着对项目的继续开发还可以在字符识别、自动化数据输入等领域有广泛应用。
- **阐释主题:** 本项目突出了计算机视觉及人工智能的主题，它利用神经网络技术，使计算机具备了模仿人类大脑的能力，使得我们可以深入研究人脑的工作方式。
- 注：在初次使用模型时需要进行几轮测试，让模型熟悉要输出的数字，再使用模型进行预测，以便模型准确率的提高。

## 3. 说明

- **编码技巧:** 在这个项目中，使用了Python编程语言和一些库，如NumPy,Tkinter和math，编写了神经网络模型，包括前向传播和反向传播算法，以实现手写数字的预测，在此之外，使用了Tkinter库来创建用户界面，允许用户绘制手写数字并进行预测。
- **逻辑流程图:** 

```
用户界面 -> 绘制手写数字 -> 图像处理 -> 预测数字 -> 显示预测结果
                           |
                      参数初始化
                           |
                      梯度下降训练
```

*流程图*

![单隐层前馈神经网络-CSDN博客](https://p.sda1.dev/13/26b4a0c4c67ee0e09d59ddb611e1d0e4/20181104123318478.png)

*神经网络工作示意图*

## 4. 参考与引用说明

- 本项目借鉴了[WYPetuous/-MNIST-: 手写数字识别+MNIST数据库：Pytorch+python (github.com)](https://github.com/WYPetuous/-MNIST-)
