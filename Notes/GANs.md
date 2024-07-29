# Generative Adversarial Nets

## Introduction

Generator 是一个可以将来自于简单分布（低维向量）的样本转换至复杂分布（高维向量）的神经网络，若我们在随机样本中添加额外的向量，此时被称为条件式生成，而 Discriminator 则是一个可以分辨样本是真实的（来自数据集中）还是生成的（来自 Generator 中）的神经网络

## Algorithm

- 初始化 Generator 和 Discriminator
- 在每一个训练循环中
  - 锁定 Generator 优化 Discriminator
  - 锁定 Discriminator 优化 Generator

## Theory behind GAN

Generator 接受从正太分布（或另外的简单的分布）采样出来的数据，并将它们转换至另一个分布 $P_{G}$ 中，而我们的目标是 $P_{G}$ 和目标数据集的分布 $P_{data}$ 尽可能的相近，如果将此视为优化问题的话，那么我们的目标就是减小 $P_{G}$ 和 $P_{data}$ 的差距，即如下公式：
$$
G^{*} = arg\min_{G}Div(P_{G},P_{data})
$$
不过我们似乎没有办法计算两个连公式都不知道的分布（至少 $P_{data}$ 的分布我们不知道吧）之间的差异，实际上，在 GANs 中的 Discriminator 就是在求解二者之间的差异，公式如下：
$$
D^{*} = arg\max_{D}V(D,G)
$$

$$
V(D,G) = E_{y \sim P_{data}}[logD(y)] + E_{y \sim P_{G}}[log(1-D(y))]
$$

上述公式中的 $V(D,G)$ 表示二元分类的损失，将来自于 $P_{data}$ 的样本划分为一类，将来自于 $P_{G}$ 的样本划分为一类，也就是我们希望来自于 $P_{data}$ 的样本有较高的得分，而来自于 $P_{G}$ 的样本有较低的得分

> __最后通过优化方法求解出来的结果和利用公式求解不同分布之间差异的结果是相关的__，在 2 个不同的分布很相似时，最大值相对较小，在不相似时，最大值将很大

综上所述，将二者相结合就可以得到如下公式：
$$
G^{*} = arg\min_{G}(\max_{D}V(G, D))
$$
也就是说，我们想让 Generator 最小化 $P_{G}$ 和 $P_{data}$ 的差距，而此差距由 Discriminator 相关的优化问题计算出来，此优化问题即如何提升 Discriminator 分辨真假（分类）的能力
