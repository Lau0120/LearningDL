# Very Deep Convolutional Networks for Large-Scale Image Recognition

## Abstract

- the effect of the convolutional network depth on its accuracy

- the large-scale image recognition setting
- using very small (3 by 3) kernels
- a significant improvement can be achieved by pushing the depth to 16-19 weight layers

## 1. Introduction

- focus on depth of ConvNet
- its feasible due to the use of very small kernels in all layers

## 2. ConvNet Configurations

### 2.1. Architecture

- input image & pre-processing

  - subtract the mean of RGB value
  - fixed-size 224 by 224 RGB image
    - randomly cropped from rescaled (S) training images (single-scale or multi-scale)
  - random hroizontal flipping & random RGB color shift

- a stack of convolutional(conv.) layers

  - a very small receptive field: 3 by 3

  - also use 1 by 1 kernels

    > which can be seen as a linear transformation of the input channels

  - stride is fixed to 1 pixel

  - the padding is 1 pixel for 3 by 3 conv. layers

    > the spatial resolution is perserved after convolution

  - 5 max-pooling layers (2 by 2 pixel window with stride 2) follow some of the conv. layers

- 3 fully-connected (FC) & final layer

  - the first two have 4096 channels
  - the third layer contains 1000 channels
  - the final layer is the soft-max layer

- misc

  - all hidden layers are equipped with ReLU
  - none of our networks contain LRN

### 2.2. Configurations

all nets differ only in the depth

... (tables) ...

> the number of weights in our nets is not greater than the number of weights in a more shallow net with larger conv. layer widths & receptive fields

### 2.3. Discussion

- using a stack of three 3 by 3 conv. layers instead of a single 7 by 7 layer

  - make the decision function more discriminative
  - decrease the number of parameters

  > this can be seen as imposing a regularisation on the 7 by 7 conv. filters, forcing them to have a decomposition through the 3 by 3 filters (with non-linearity injected in between)

- the incorporation of 1 by 1 conv. layers is a way to increase the non-linearity of the decision function without affecting the receptive fields of the conv. layers

## 3. Classification Framework

### 3.1. Training

the training is carried out by optimising the multinomial logistic regression objective using mini-batch gradient descent with momentum

- batch size: 256
- momentum: 0.9
- weight decay: 0.0005
- dropout ratio: 0.5
  - the first two fully-connected layers
- learning rate: 0.01 (init)
  - decreased by a factor of 10, 3 times in total
- epochs number: 74
- weights: normal distribution
  - zero mean
  - 0.01 variance
- biases: 0

### 3.2. Testing

- input images & pre-processing
  - rescaled images Q (is not necessarily equal to the training scale S)
  - resize to 224 by 224
  - horizontal flipping
- the fully-connected layers are first converted to convolutional layers
  - the first FC layer to a 7 by 7 conv. layer
  - the last 2 FC layers ot 1 by 1 conv. layer

### 3.3. Implementation Details

multi-GPU training is carried out by splitting each batch of training images into several GPU batches, processed in parallel on each GPU

after the GPU batch gradients are computed, they are averaged to obtain the gradient of the full batch

## 4. Conclusion

the representation depth is beneficial for the classification accuracy
