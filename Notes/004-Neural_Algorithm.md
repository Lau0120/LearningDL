# A Neural Algorithm of Artistic Style

- an artificial system based on a Deep Neural Network that creates artistic images of high perceptual quality

  - the system uses neural representations to separate & recombine content and style of arbitrary images

- convolutional neural networks

  - each filters of layers __extracts a certain feature from the input image (feature maps)__
  - they develop a representation of the image that _makes object information increasingly explicit along the processing hierarchy_
    - along the processing hierarchy of the network, the input image is transformed into representataions that _increasingly care about the __actual content__ of the image compared to its detailed pixel values_

- image visualization

  - higher layers in the network capture the high-level content in terms of objects & their arrangement in the input image
  - reconstructions from the lower layers simply reproduce the exact pixel values of the original image

  > we therefore refer to the feature responses in higher layers of the network as the _content representation_

- use a __feature space__ designed to capture texture information

  - built on top of the filter responses in each layer of the network
  - consists of the correlations between the different filter responses over the spatial extent of the feature maps

- reconstructions from the style features produce texturised versions of the input image that capture its general appearance in terms of color & localised structures

  - refer to this multi-scale representation as _style representation_
    - the style representation is a multi-scale representation that includes multiple layers of the neural network

  > include layers from the whole network hierarchy or only a smaller number of lower layers (the most appealing images are usually created by matching the style representation up to the highest layers in the network)

- representations of content & style in the CNN are separable

  - so we can manipulate both representations independently to produce new meaningful images
  - generate images that mix the content & style representation from 2 different source images
    - __by finding an image that simultaneously matches the content representation & the style representation__

- loss function

  - contains 2 terms for content and style respectively
  - for a specific pair of source images one can adjust the trade-off between content and style to create visually appealing images

## Methods

- on the basis of the __VGG-Network__
  - 19 layers
  - without any connected layers
  - average pooling

> in general, we minimize __style loss__ from some lower layers & __content loss__ from the higher layer  _simultaneously_