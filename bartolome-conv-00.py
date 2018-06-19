from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math

image_X = 1024
image_Y = 1024

cell_X = 256
cell_Y = 256

stride = 8

frames_X = image_X / stride
frames_Y = image_Y / stride

# test flag for batch norm
bnflag = tf.placeholder(tf.bool)
bniter = tf.placeholder(tf.int32)


print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# input array
X = tf.placeholder(tf.float32, [None, 1024, 1024, 3])

# output should be an array of cell-size images.
# Bit depth can be basically 1bpp or 2bpp if we feel fancy
Y = tf.placeholder(tf.int8, [None, frames_X, frames_Y, cell_X, cell_Y, 1])

def convLayer(input, patchdim, indepth, outdepth):
    W = tf.Variable(tf.truncated_normal([patchdim, patchdim, indepth, outdepth]))
    B = tf.Variable(tf.Constant(0.1, tf.float32, [outdepth]))
    Yl = tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding="SAME")
    