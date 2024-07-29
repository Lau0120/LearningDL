# Deep Residual Learning for Image Recognition

## Abstract

- reformulate the layers as _learning residual functions_ with _reference to the layer inputs_, instead of learning unreferenced functions
- residual networks are easier to optimize & gain accuracy from considerably increased depth
- present analysis on CIFAR-10 with 100 & 1000 layers
- the depth of representations is of central importance for many visual recognition tasks

## 1. Introduction

- with the network depth increasing, accuracy gets saturated and then degrades rapidly
  - (unexpectedly), such degradation is not caused by overfitting
  - _stacking more layers to a deep model leads to higher training error_
- a deep residual learning framework
  - we explicitly let these layers (to fit underlying mapping) fit a residual mapping：$F(x) + x$
  - the residual mapping can be realized by deedforward neural networks with _shortcut connections_ (skipping one or more layers)
    - the shortcut connections __simply perform identity mapping__
    - their outputs are __added to the outputs of the stacked layers__
- advantages
  - easy to optimize
  - easily enjoy accuracy gains from greatly increased depth

## 3. Deep Residual Learning

### 3.1. Residual Learning

...

### 3.2. Identity Mapping by Shortcuts

a building block defined as:
$$
y = F(x, \{W_i\}) + x
$$

- $F + x$ is performed by a shortcut connection & __element-wise addition__
- apply the __last nonlinearity after the addition__
- the dimensions of $x$ and $F$ must be equal in this function (mapping)
  - we can perform a linear projection matrix to match the dimensions
    - _the identity mapping is sufficient for addressing the degradation problem & is economical_
    - only used when matching dimensions
- the residual function $F$ is flexible
  - more layers are possible
  - similar to a linear layer while $F$ has only a single layer
- the function is applicable to convolutional layers

### 3.3. Network Architectures

2 models for ImageNet

- Plain Network

  - inspired by philosophy of VGG nets
    - convolutional layers mostly have 3 by 3 filters\
    - for the same output feature map size, the layers have the same number of filters
    - if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer
  - maxpool layers have a stride of 2
  - the network ends with a global average pooling layer
  - a 1000-way fully connected layer with softmax

- Residual Network

  - based on above palin network, we insert shortcut connections

  - the identity shortcuts can be directly used when input & output are of the same dimensions

  - when the dimensions increase, there are 2 options:

    - (no extra parameter) still performs identity mapping, with extra zero entries padded for increasing dimensions
    - using 1 by 1 convolutions to match dimensions

    > for both options, __when the shortcuts go across feature maps of 2 sizes, they are performed with a stride of 2__

### 3.4. Implementation

- input images
  - resize randomly in [256, 480]
  - randomly crop to 224 by 224
  - horizontal flip
  - per-pixel mean subtracted
  - standard color augmentation
- batch normalization
  - right after each convolution & before activation
- weights init: ...
- batch size: 256
- epoch counts: 600000
- weight decay: 0.0001
- momentum: 0.9
- without dropout
