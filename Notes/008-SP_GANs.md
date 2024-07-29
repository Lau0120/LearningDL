# SP-GAN

## Introduction

在三维形状构建中一个颇具挑战性的问题是如何构建生成模型来生成新的、多样的且看起来真实的三维形状，现有的解决方案如下：

- 部件级别的分解和重组
- 描述形状分布的生成模型
  - 自回归模型
  - 流模型
- 基于无监督的生成式对抗网络

<blockquote alt="warn">
    <p>
        部件级别的分解和重组属于有监督学习的类型，所以依赖于部件注解的质量和可用性
    </p>
</blockquote>

由于形状的多样性和三维空间的高复杂性，上述方法存在一系列问题：

1. 生成的形状不具备良好的细节
2. 倾向于生成噪声
3. 缺乏局部感知的生成和控制

<blockquote alt="warn">
    <p>
        传统的生成方法所依赖的随机隐编码只描述了全局的形状差异，很难获取到隐编码维度和生成形状细节之间的正确关系
    </p>
</blockquote>

SP-GAN 可以解决上述的问题，即模型在可以生成具备良好细节且低噪声的形状的同时拥有干预生成过程（结果）的可控性，不同于传统卷积生成网络将单一隐编码作为输入，SP-GAN 有两个解耦的输入：

- 全局先验 $S$
  - 单位球样式的三维点云
  - 为生成过程提供无偏差（unbiased）的空间上的指导
- 局部先验 $z$
  - 随机隐编码
  - 为生成过程提供局部细节

通过将一个隐编码 $z$ 附着在 $S$ 中的每个点之后构成一个先验矩阵，即实现生成过程从一个共享的全局初始开始，让每个点向它所预期的位置移动从而生成模型，关键在于将生成问题分解为两个子问题来可以减缓模型的学习曲线且增强生成形状的质量：

1. 全局的形状生成
2. 局部的机构调整

通过嵌入单位球和生成形状之间的隐式稠密关系，SP-GAN 可以干预模型的生成过程，形象地说，单位球提供了形状生成和控制的工作空间（类似于一个用于绘画的画布），在单位球的某个特定区域"绘画"时将自然地控制不同生成形状的相应区域，也就是说，如果我们编辑了和 $S$ 中某个特定点所关联的隐向量，我们将控制形状中与之关联的区域的局部结构

<blockquote alt="info">
    <p>
        由于局部区域之间几何和语义上的关联，在编辑某个区域时，有可能轻微地编辑另外一些区域
    </p>
</blockquote>
## Overview

![SP-GANs_General](C:\Users\vcc\Documents\Classic_Papers\Point_Cloud\SP_GANs\SP-GANs_General.png)

以点云的形式表示三维形状，模型包含 2 个解耦的输入

1. 全局先验 $S$：从单位球上采样的 $N$ 个三维坐标点
2. 局部先验 $z$：从标准正太分布采样的 $d$ 维随机隐向量

通过将隐编码附着在全局先验的每个点之后，从而将这两种先验组合为一个先验矩阵

在训练时，生成器从 $S$ 和 $z$ 生成点云 $\Rho$，再从给定的三维形状中采样和生成点云个数以及维度都相同的真实点云 $\hat{\Rho}$

- 让判别器学习二者之间的差异

- 让生成器学习如何生成以假乱真的点云

## Method

### Generator

![SP-GANs_Generator](C:\Users\vcc\Documents\Classic_Papers\Point_Cloud\SP_GANs\SP-GANs_Generator.png)

- 隐编码 $z$ 为特征引入了各种不同的局部风格和良好细节
- 全局先验 $S$ 允许我们应用具有空间相关性的卷积来实现特征提取

<blockquote alt="success">
    <p>
        传统模型将单个隐编码作为输入，在早期阶段必须利用全连接网络来实现特征提取，如此以来需要相对大量的可学习参数，同时还限制了模型的表现力
    </p>
</blockquote>

- 利用 Adain 来将结构差异和局部风格嵌入到特征图中

<blockquote alt="info">
    <p>
        具体来说，就是将特征所在的分布通过目标的均值和方差迁移到目标分布上，目标分布的均值和方差来自于特征嵌入（Feature Embedding）和风格嵌入（Style Embedding）结果
    </p>
</blockquote>

- 最后的结构则类似于 PointNet 的方法，利用最大池化函数抽取全局特征之后，将它和局部特征组合起来得到最终的生成点云

#### Graph Attention Module

![SP-GANs_GAM](C:\Users\vcc\Documents\Classic_Papers\Point_Cloud\SP_GANs\SP-GANs_GAM.png)

GAM（Graph Attention Module）采用了 DGCNN 的基本结构且有一些简单的调整来进一步考虑特征空间中 $K$ 个邻居之间的关系

### Discriminator

![SP-GANs_Discriminator](C:\Users\vcc\Documents\Classic_Papers\Point_Cloud\SP_GANs\SP-GANs_Discriminator.png)

- 判别器可以学习利用表示整个形状的全局特征向量来预测点云的来源

- 考察局部细节也可以作为判别点云真实性的依据，所以额外地添加了和每个点相关的分数

<blockquote alt="info">
    <p>
        判别器中的特征提取网络以 PointNet 为基础
    </p>
</blockquote>

如此以来判别器就可以同时考虑之于输入来说相关的全局和局部的差异，以此为引导让生成器考虑全局结构的同时也必须考虑局部的细节

### Details

训练时，判别器的损失 $\mathcal{L}_{D}$ 如下所示
$$
\mathcal{L}_{D} = \mathcal{L}_{D}^{shape} + \lambda\mathcal{L}_{D}^{point}
$$
全局结构损失 $\mathcal{L}_{D}^{shape}$ 如下所示：
$$
\mathcal{L}_{D}^{shape} = \frac{1}{2}[(D(\Rho) - 0)^2 + (D(\hat{\Rho}) - 1)^2]
$$
局部细节损失 $\mathcal{L}_{D}^{point}$ 如下所示：
$$
\mathcal{L}_{D}^{point} = \frac{1}{2N}\sum_{i = 1}^{N}[(D(\rho_{i}) - 0)^2 + (D(\hat{\rho}_{i}) - 1)^2]
$$
训练时，生成器的损失 $\mathcal{L}_{G}$ 如下所示：
$$
\mathcal{L}_{G} = \frac{1}{2}[D(\Rho) - 1]^{2} + \beta\frac{1}{2N}\sum_{i = 1}^{N}[D(\rho_{i}) - 1]^2
$$
以下是训练时的超参数以及注意点：

1. 采用 Adam 进行优化
2. 网络的学习率都是 0.0001
3. 网络都用 LeakyReLU 作为非线性激活
4. 生成器最后一层的激活函数是 Tanh
5. 邻域个数 K 是 20

必须注意的是：

- 在训练阶段，在每次训练迭代中，必须保持全局先验  $S$ 不变，而隐编码 $z$ 可以从标准正态分布中随机采样

- 在推理阶段，全局先验和训练时所采用的相同，局部先验随机采样，随着采样隐编码的不同，生成器将生成各种各样的点云