# ImageNet Classification with Deep Convolutional Neural Networks

## Abstract

- ImageNet LSVRC-2010 Contest	
- 1000 different classes
- better than previous state-of-the-art
- consists of five convolutional layers, some of which are followed by max-pooling layers & three fully connected layers with a final 1000-way softmax
- used non-saturating neurons a very efficient GPU implementation of the convolution operation
- employed dropout to reduce overfitting

## 1. Prologue

With enough computation & enough data, learning beats programming for complicated tasks that require the integration

SuperVision evolved from the multilayer neural networks that were widely investigated in the 1980s

- backpropagation
- more labeled data & more computation

> how the classification performance of the whole network depended on the value of the weight on each connection

## 2. Introduction

...

## 3. The Dataset (Pre Processing)

- requires a constant input dimensionality
- down-sampled the images to a fixed resolution of 256 by 256, given a rectangular image
  - rescaled the image such that the shorter side was of length 256
  - then cropped out the central 256 by 256 patch from the resulting image
- subtracting the mean activity over the training set from each pixel

## 4. The Architecture

### 4.1. Rectified Linear Unit (ReLU) nonlinearty

- in terms of training time with gradient descent, these <u>saturating</u> nonlinearities are much slower than the nonlinearity
- Deep CNNs with ReLUs train serveral times faster than their equivalents with tanh units
- the accelerated ability to fit the training set with ReLUs

### 4.2. Training on multiple GPUs

- spread the net across 2 GPUs
- puts half of the neurons on each GPU

### 4.3. Local response normalization

- local normalization scheme aids generalization
  $$
  b_{x,y}^{i} = a_{x,y}^{i} / (k + \alpha\sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)}(a_{x,y}^{j})^{2})^{\beta}
  $$

  > the $i^{th}$ kernel activity is divided correspondingly by the activities of serveral kernels around it

  - $k = 2$
  - $n = 5$
  - $\alpha = 10^{-4}$
  - $\beta = 0.75$
  
- applied this after applying the ReLU nonlinearity in certain layers

- this sort of response normalization implements a form of lateral inhibition inspired by the type found in real neurons

### 4.4. Overlapping pooling

- stride is 2, size is 3

- produces output of equivalent dimensions
- overlapping pooling is more difficult to overfit

### 4.5. Overall architecture

- the first 5 are convolutional
  - Block1
    - Conv2d
    - ReLU
    - LRN
    - MaxPool
  - Block2
    - Conv2d
    - ReLU
    - LRN
    - MaxPool
  - Block3
    - Conv2d
    - ReLU
  - Block4
    - Conv2d
    - ReLU
  - Block5
    - Conv2d
    - ReLU
    - MaxPool
- the reamining 3 are fully connected
  - Block6
    - Conv2d
    - ReLU
    - Dropout
  - Block7
    - Linear
    - ReLU
    - Dropout
  - Block8
    - Linear
    - Softmax

> Max-pooling layers follow both response-normalization layers as well as the fifth convolutional layer. The ReLU non linearity is applied to the output of every convolutional and fully connected layer

## 5. Reducing Overfitting

Its insufficient to learn so many parameters without considerable overfitting

### 5.1. Data augmentation

transform images in Python code on the CPU, while the GPU is training on the previous batch of images

- generating image translations & horizontal reflections

  - training

    - crop it from 256 (reszied) to 224 (227)

  - validation

    - prediction by extracting ten 224 patches (4 corner & 1 center) in front & back (flipped horizontally)
    - averaging the predictions made by the network's softmax layer on the 10 patches

    > it means that you cannot use default sampler on dataloader in PyTorch

- altering the intensities of the RGB channels in training images

  - perform PCA on the set of RGB pixel values throughout the ImageNet training set

### 5.2. Dropout

combining the predictions of many different models is a very successful way to reduce test errors

- training

  - the neurons "dropped out" in this way do not contribute to the forward pass and participate in back propagation

  - every time and input is presented, the neural network samples a different architecture

  - but all these architectures share weights

- validation

  - use all the neurons but multiply their outputs by 0.5

## 6. Details of Learning

optimizer is momentum

- training
  - batch size: 128
  - momentum: 0.9
  - weight decay: 0.0005 (important for the model to learn)
  - weight (init): a zero-mean Gaussian distribution with standard deviation 0.01
  - bias (init): 
    - 2, 4 & 5 convolutional layers as well as in the fully connected hidden layers with the constant 1
    - the remaining layers with the constant 0
  - learninga rate (init): 0.01
    - divide the learning rate by 10 when the validation error rate stopped improving
    - reduced 3 times prior to termination
  - epochs count: roughly 90 cycles

## 7. Results

- Top-1 (val): 40.7
- Top-5 (val): 18.2

## Supplementary Knowledge

- saturating / non-saturating
