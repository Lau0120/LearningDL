# Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization

## Abstract

- original fream-work requires a slow iterative optimization process
  - which limits its practical application
- feed-forward neural networks have been proposed to speed up neural style transfer
  - the network is usually tied to a fixed set of styles and cannot adapt to arbitrary new styles
- present a simple yet effective approach that for the first time enables arbitrary style transfer in real-time
  - adaptive instance normalization (AdaIN) layer
    - aligns the mean and variance of the content features with those of the style features
- advantages
  - comparable to the fastest existing approach
  - without the restriction to a pre-defined set of styles
  - allows flexible user controls such as content-style trade-off, style interpolation, color & spatial controls
  - all using a single feed-forward neural network

## Introduction

our approach can transfer arbitrary new styles in real-time, combining the flexibility of the optimization-based framework and the speed similar to the fastest feed-forward approaches

propose a new interpretation that instance normalization performs style normalization by normalizing feature statistics, which have been found to carry the style information of an image

we introduce a simple extension to IN, namely adaptive instance normalization (AdaIN)

given a content input and a style input, AdaIN simply adjusts the mean and variance of the content input to match those of the style input

we find AdaIN effectively combines the content of the former and the style latter by transferring feature statistics

A decoder network is then learned to generate the final stylized image by inverting the AdaIN output back to image space

## Background

- Batch Normalization
  - there is discrepancy between training (mini-batch statistics) and inference (popular statistics)
  - BN can alleviate domain shifts by recomputing popular statistics in the target domain
- Instance Normalization
  - significant improvement could be achieved simply by replacing BN layers with IN layers
  - IN layers are applied at test time unchanged
- Conditional Instance Normalization
  - learns a different set of parameters for each style
  - a style image with its index are randomly chosen from a fixed set of styles

## Interpreting Instance Normalization

- IN takes place in the feature space, therefore it should have more profound impacts that a simple contrast normalization in the pixel space
  - the affine parameters in IN can completely change the style of the output image
- matching many other statistics, including channel-wise mean and variance, are also effective for style transfer
  - instance normalization performs a form of style normalization by normalizing feature statistics, namely the mean and variance
- the feature statistics of a generator network can also control the style of the generated image
  - original images
    - model with IN converges faster than model with BN
  - contrast normalized images
    - as same as original images
  - style normalized (Style Transfer Net)
  
    - model with IN converges still faster than model with BN, but the gap is smaller than before
  
      - the speed of IN model is slightly faster than before
  
      - the speed of BN model is even faster than before
  
    > style normalization works well, IN does perform a kind of style normalization

IN can normalize the style of each individual sample to the target network, training is facilitated because the rest of the network can focus on content manipulation while discarding the original style information

## Adaptive Instance Normalization

if IN normalizes the input to a single style specified by the affine parameters (CIN), is it possible to adapt it to arbitrarily given styles by using adaptive affine transformations (AdaIN)

AdaIN receives a content input $x$ and a style input $y$, and simply aligns the channel wise mean and variance of $x$ to match those of $y$

AdaIN has no learnable affine parameters, it adaptively computes the affine parameters from the style input:
$$
AdaIN(x, y) = \sigma(y)(\frac{x - \mu(x)}{\sigma(x)}) + \mu(y)
$$
in which we simply scale the normalized content input with $\sigma(y)$, and shift it with $\mu(y)$

> AdaIN (no computational cost) performs style trasfer in the feature space by transferring feature statistics, specifically the channel-wise mean and variance
