########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
#from imagenet_classes import class_names
import matplotlib.pyplot as plt


###############################CLASS###########################################################
class vgg16:
    def __init__(self, imgs, weights=None, sess=None):  # Intitiate network graph and load weight
        self.imgs = imgs
        self.convlayers()  # build convolutional layers of vgg network

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)  # Load network weight

    def convlayers(self):  # build convolutional layers of vgg network
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

    def load_weights(self, weight_file, sess):  # Load network weight
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if (i < len(self.parameters)):
                print(i, k, np.shape(weights[k]), len(self.parameters));
                sess.run(self.parameters[i].assign(weights[k]))


sess = tf.Session()
img1 = imread('roof5.jpg', mode='RGB')  # Load image
Sy, Sx, dpt = img1.shape  # Get image shape
img1 = imresize(img1, (300, int(300 / Sy * Sx)))
Sy, Sx, dpt = img1.shape  # Get image shape
imgs = tf.placeholder(tf.float32, [None, Sy, Sx, 3])
vgg = vgg16(imgs, 'vgg16_weights.npz', sess)  # Initiate neural network

conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3 = sess.run(
    [vgg.conv1_1, vgg.conv1_2, vgg.conv2_1, vgg.conv2_2, vgg.conv3_1, vgg.conv3_2, vgg.conv3_3, vgg.conv4_1,
     vgg.conv4_2, vgg.conv4_3, vgg.conv5_1, vgg.conv5_2, vgg.conv5_3], feed_dict={vgg.imgs: [img1]})

from scipy import signal
from scipy import misc


def DeepSobel(
        im):  # Apply sobel oprator of response map of specific layer in the net to get its total gradient/edge map of this layer
    im = im.squeeze()  # Remove acces dimension
    im = np.swapaxes(im, 0, 2)  # Swap axes to feet thos of standart image (x,y,d)
    im = np.swapaxes(im, 1, 2)
    Gx = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # Build sobel x,y gradient filters
    Gy = np.swapaxes(Gx, 0, 1)  # Build sobel x,y gradient filters
    ndim = im[:, 1, 1].shape[0]  # Get the depth (number of filter of the layer)
    TotGrad = np.zeros(im[1, :, :].shape)  # The averge gradient map of the image to be filled later

    for ii in range(ndim):  # Go over all dimensions (filters)
        # print(ii);
        gradx = signal.convolve2d(im[ii, :, :], Gx, boundary='symm', mode='same');  # Get x sobel response of ii layer
        grady = signal.convolve2d(im[ii, :, :], Gy, boundary='symm', mode='same');  # Get y sobel response of ii layer
        grad = np.sqrt(np.power(gradx, 2) + np.power(grady, 2));  # Get total sobel response of ii layer
        TotGrad += grad  # Add add to the layer average gradient/edge map
    TotGrad /= ndim  # Get layer sobel gradient map
    return TotGrad


def SSobel(im, sp):
    TotGrad = DeepSobel(im)
    NewGrad = misc.imresize(TotGrad, sp, interp='bicubic')  # (‘nearest’,‘bilinear’ , ‘bicubic’ or ‘cubic’)
    print("New Size", NewGrad.shape)
    return NewGrad


def to_rgb1(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


from scipy import signal
from scipy import misc


def WriteSobelResultsforAllFilters(im, LayerName,
                                   in_im):  # Apply sobel oprator of response map to all filters respones map in given layer and display the results
    sp = in_im[:, :, 1].shape

    sp = list(sp)
    sp[0] *= 2
    im = im.squeeze()  # Remove acces dimension
    im = np.swapaxes(im, 0, 2)  # Swap axes to feet thos of standart image (x,y,d)
    im = np.swapaxes(im, 1, 2)
    Gx = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # Build sobel x,y gradient filters
    Gy = np.swapaxes(Gx, 0, 1)  # Build sobel x,y gradient filters
    ndim = im[:, 1, 1].shape[0]  # Get the depth (number of filter of the layer)
    TotGrad = np.zeros(im[1, :, :].shape)  # The averge gradient map of the image to be filled later


    for ii in range(ndim):  # Go over all dimensions (filters)
        print(LayerName, '    Filter: ', ii + 1);
        gradx = signal.convolve2d(im[ii, :, :], Gx, boundary='symm', mode='same');  # Get x sobel response of ii layer
        grady = signal.convolve2d(im[ii, :, :], Gy, boundary='symm', mode='same');  # Get y sobel response of ii layer
        grad = np.sqrt(np.power(gradx, 2) + np.power(grady, 2));  # Get total sobel response of ii layer
        FileName = LayerName + " Filter " + str(ii) + ".png";
        fim = np.concatenate((im[ii, :, :], grad), axis=0)

        fim = misc.imresize(fim, sp, interp='bicubic')  # (‘nearest’,‘bilinear’ , ‘bicubic’ or ‘cubic’)
        fim = to_rgb1(fim);
        fim = np.concatenate((in_im, fim), axis=0)

        plt.imshow(fim)  # Display results for filter
        plt.gray()
        plt.show()
        plt.imsave(FileName, fim)



print('Display sobel gradient map for all filters in all layers')
# ................Original image........................................................
print('Input image')
plt.imshow(img1)
plt.gray()
plt.show()
#plt.show(TotGrad)
# ....................11................................................................
print('Layer Name: conv1_1')
WriteSobelResultsforAllFilters(conv1_1, 'F:', img1)
#C:/Users/Sagi Eppel/Desktop/VGG_RESULTS/conv1_1

# ....................12................................................................
print('Layer Name: conv1_2')
WriteSobelResultsforAllFilters(conv1_2, 'F:', img1)
# ....................21................................................................
print('Layer Name: conv2_1')
WriteSobelResultsforAllFilters(conv2_1, 'F:', img1)
# ....................22................................................................
print('Layer Name: conv2_2')
WriteSobelResultsforAllFilters(conv2_2, 'F:', img1)
# ....................31................................................................
print('Layer Name: conv3_1')
WriteSobelResultsforAllFilters(conv3_1, 'F:', img1)
# ....................32................................................................
print('Layer Name: conv3_2')
WriteSobelResultsforAllFilters(conv3_2, 'F:', img1)
# ....................33................................................................
print('Layer Name: conv3_3')
WriteSobelResultsforAllFilters(conv3_3, 'F:', img1)
# ....................41................................................................
print('Layer Name: conv4_1')
WriteSobelResultsforAllFilters(conv4_1, 'F:', img1)
# ....................42................................................................
print('Layer Name: conv4_2')
WriteSobelResultsforAllFilters(conv4_2, 'F:', img1)
# ....................43................................................................
print('Layer Name: conv4_3')
WriteSobelResultsforAllFilters(conv4_3, 'F:', img1)
# ....................51................................................................
print('Layer Name: conv5_1')
WriteSobelResultsforAllFilters(conv5_1, 'F:', img1)
# ....................52................................................................
print('Layer Name: conv5_2')
WriteSobelResultsforAllFilters(conv5_2, 'F:', img1)
# ....................53................................................................
print('Layer Name: conv5_3')
WriteSobelResultsforAllFilters(conv5_3, 'F:', img1)