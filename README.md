#Edge-Detection-Using-Deep-Learning-(CNN)

This repository is my implementation of a project that uses CNN to detect edges of 2-D Roof images.

Traditional methods for edge detection like filters or derivatives are not powerful enough to learn mappings from image patches for edge predictions so I am incorporating CNN into edge detection problem.

The methodology comprises of four major steps:
a. Find the feature map of the of the image by applying the convolutional layers of the network. 
b. Apply sobel operator to  get gradient map for each filter in a given convolutional layer.
c. Take the average of response map of all filters in a given layer to get the total edge map of a specific layer.
d. Resize the edge map to the original image size to display results

#Description
This code uses Tensorflow implementation of VGG network by David Frossard, 2016 to compute feature map of images. To find the feature maps of images you first need to download the pretrained model weights from here: http://www.cs.toronto.edu/~frossard/post/vgg16/ and put them in the code folder. The response of convolutional neural network in each layer is a matrix of size Dx*Dy*d were d is the number of filters in this layer (can be considered as the layer depth). Dx and Dy are is the x,y dimension of the layer and is some fraction of the input image dimensions. By using the the response map of a given filter in in given layer (after RELU) as 2d image of dimension (Dx,Dy,1) and applying the sobel operator on this image, it possible to get the gradient/edge map of this filter in this layer. Summing the absolute value of the sobel operator of all feature in an image give the Total gradient/edge map for this layer , which seem to be rather similar to gradient map of input image.

#Dependencies 
The Convolutional Nueral Network used here is the VGG implementation for Tensorflow by David Frossard. The weight for this pre-trained network should be download from here:http://www.cs.toronto.edu/~frossard/post/vgg16/ and put in code directory.

Python 3.4.5 

Tensorflow





