# Project Title
**Edge Detection Using CNN (Deep Learning)**

## Author
* **Name:** Aneeqa Ahmed
* **Country Name:** Pakistan
* **Education :** Graduate Student Jeju National University, Korea
* **Mentor:** Dr. Yungcheol Byun
* **Motivation:** Currently there is not a single detector with both efficiency and reliability. Traditional filter based algorithms have the advantage of theoretical strictness but need extensive post processing. Existing methods usually rely. heavily on computing multiple image features, which makes the whole system complex and computationally expensive. Traditional methods for edge detection like filters or derivatives are not powerful enough to learn mappings from image patches for edge predictions so I decided incorporate CNN into edge detection problem

## Introduction

This repository is my implementation of a project that uses CNN to detect edges of 2-D Roof images.
Traditional methods for edge detection like filters or derivatives are not powerful enough to learn mappings from image patches for edge predictions so I am incorporating CNN into edge detection problem.

The methodology comprises of four major steps:
* Find the feature map of the of the image by applying the convolutional layers of the network 
* Apply sobel operator to  get gradient map for each filter in a given convolutional layer
* Take the average of response map of all filters in a given layer to get the total edge map of a specific layer
* Resize the edge map to the original image size to display results

## Flow Diagram

![ALT text](/FlowDiagram.png "Flow Diagram") 


## Results
* VGG network is characterized by its simplicity, using only 3×3 convolutional layers stacked on top of each other in increasing depth
* VGGNet consists of 16 convolutional layers and is very appealing because of its very uniform architecture
* Trained on 4 GPUs for 2–3 weeks. It is currently one of the preferred choice for extracting features from images

### Gradient Map-ALL Filters
![ALT text](/Gradient-Map-ALL-Filters.png "Gradient Map-ALL Filters")

### Total Edge Map of ALL Layers
![ALT text](/Total-Edge-Map-of-all-layers.png "Total Edge Map of all layers")

### Averaged ALL layers Response Map Outcome
![ALT text](/Averaged-all-layers-response-map-outcome.png "Averaged all layers response map outcome")

## Description

This code uses Tensorflow implementation of VGG network by David Frossard, 2016 to compute feature map of images.To find the feature maps of images you first need to download the pretrained model weights and put them in the code folder. The response of convolutional neural network in each layer is a matrix of size Dx*Dy*d were d is the number of filters in this layer (can be considered as the layer depth). Dx and Dy are the x,y dimension of the layer. By using the the response map of a given filter in in given layer (after RELU) as 2d image of dimension (Dx,Dy,1) and applying the sobel operator on this image, it possible to get the gradient/edge map of this filter in this layer. Summing the absolute value of the sobel operator of all feature in an image give the Total gradient/edge map for this layer , which seem to be rather similar to gradient map of input image.

## Dependencies 
* The Convolutional Nueral Network used here is the VGG implementation for Tensorflow by David Frossard. The weight for this pre-trained network should be downloaded from [here](http://www.cs.toronto.edu/~frossard/post/vgg16/) and then put it in code directory to make the code work.
* Python 3.4.5 
* Tensorflow

## Acknowledgments

```
* Asian Machine Learning Camp, Jeju National University
```

![ALT text](/camp_member.png "Camp Member")
