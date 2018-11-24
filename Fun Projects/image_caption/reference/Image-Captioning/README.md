# Image Captioning

This repository contains an implementation of image captioning based on neural network (i.e. CNN + RNN). The model first extracts the image feature by CNN and then generates captions by RNN. CNN is VGG16 and RNN is a standard LSTM .

Normal Sampling and Beam Search were used to predict the caption of images.


# Network Topology:-

## Encoder
The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.
## Decoder
The Decoder is a Recurrent Neural Network(RNN) which does language modelling up to the word level. The first time step receives the encoded output from the encoder and also the <START> vector.

Dataset used was <a href="http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html">Flickr8k dataset</a>.

# Input
![input](https://user-images.githubusercontent.com/23000971/33495332-fbd2b75a-d6eb-11e7-999a-09fdc4255a6f.JPG)


# Output
![output](https://user-images.githubusercontent.com/23000971/33495366-2b5a9cd6-d6ec-11e7-9cd0-2b7adce57b3e.JPG)
![text](https://user-images.githubusercontent.com/23000971/33495435-7a9bd10c-d6ec-11e7-9b26-77c6865c0551.JPG)


# Dependencies

* Keras 2.0.7
* Theano 0.9.0
* Numpy
* Pandas 0.20.3
* Matplotlib
* Pickle

# References

[1] Deep Visual-Semantic Alignments for Generating Image
Descriptions ( Karpathy et-al, CVPR 2015) 

[2] Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan <a href="https://arxiv.org/abs/1411.4555">Show and Tell: A Neural Image Caption Generator</a>

[3] CS231n: Convolutional Neural Networks for Visual Recognition.
( Instructors : Li Fei Fei, Andrej Karpathy, Justin Johnson)
