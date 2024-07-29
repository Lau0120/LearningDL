# Unsupervised Representation Learning with Deep Convolutional GANs

## Abstract

- to help bridge the gap between the success of CNNs for supervised learning & unsupervised learning
- DCGANs have certain architectural constraints & they are a strong candidate for unsupervised learning

## Introduction

- GANs provide an attractive alternative to maximum likelihood techniques
- propose & evaluate a set of constraints on the architectural topology of Convolutional GANs that make them stable to train in most settings

## Approach and Model Architecture

core to our approach is adopting & modifying 3 recently demonstrated changes to CNN architectures

1. replace any pooling layers with strided convolutions (discriminator) & fractional-strided convolutions (generator)
2. eliminate fully connected layers on top of convolutional features
3. use batch normalization in both the generator & the discriminator
4. use ReLU activation in generator for all layers except for the output, which uses Tanh
5. use LeakyReLU activation in the discriminator for all layers

## Details of Adversarial Training

- datasets
  - LSUN
  - ImageNet-1k
  - Faces
- pre-processing
  - scaling to the range of the tanh
- hyper-parameters
  - mini-batch size of 128 (SGD)
  - slope leak 0.2 (LeakyReLU)
  - learning rate 0.0002, $\beta$ is 0.5 (Adam)
- Initialization
  - zero-centered Normal Distribution
  - standard deviation 0.02
