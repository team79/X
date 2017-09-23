# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

from six.moves import urllib
import tarfile



from tensorflow.python.lib.io import file_io
import os
import sys
import numpy as np
import pickle
import argparse
import scipy
from skimage import io
import random
#import read_image_in_pai

BatchSize = 128
SampleLen = 632
TrainLen = 316
TestLen = 316
ImageHeight = 128
ImageWidth = 48
ImageChannel = 3
FinalLocalSize = 800

#FilePath = "G:\DML\数据库\VIPeRa\\all"

# #-----------------------------------------------------------------------------------------------------------------------------
# #--------------------------------------------  deeplearning model  -----------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------------------

print("----------------------------------------\n" * 2)
print("begin construct deep model:")

def variable_with_weight_loss( shape, stddev, w1 ):
    var = tf.Variable( tf.truncated_normal( shape, stddev = stddev ) )
    if w1 is not None:
        weight_loss = tf.multiply( tf.nn.l2_loss( var ), w1, name = 'weight_loss' )
        tf.add_to_collection( 'losses', weight_loss )
    return var

image_holder = tf.placeholder( tf.float32, [BatchSize*2, ImageHeight, ImageWidth, ImageChannel] )
label_holder = tf.placeholder( tf.int32, [BatchSize*2] )

weight1 = variable_with_weight_loss( shape = [5, 5, 3, 64 ], stddev = 5e-2, w1 = 0.0 )
kernel1 = tf.nn.conv2d( image_holder, weight1, [ 1, 1, 1, 1 ], padding = 'SAME' )
bias1 = tf.Variable( tf.constant( 0.0, shape = [64] ) )
conv1 = tf.nn.relu( tf.nn.bias_add( kernel1, bias1 ) )
pool1 = tf.nn.max_pool( conv1, ksize = [ 1, 3, 3, 1 ], strides = [ 1, 2, 2, 1 ], padding = 'SAME' )
norm1 = tf.nn.lrn( pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75 )

weight2 =variable_with_weight_loss( shape = [ 5, 5, 64, 64 ], stddev = 5e-2, w1 = 0.0 )
kernel2 = tf.nn.conv2d( norm1, weight2, [ 1, 1,  1, 1 ], padding = 'SAME' )
bias2 = tf.Variable( tf.constant( 0.1, shape = [64] ) )
conv2 = tf.nn.relu( tf.nn.bias_add( kernel2, bias2 ) )
norm2 = tf.nn.lrn( conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75 )
pool2 = tf.nn.max_pool( norm2, ksize = [ 1, 3, 3, 1 ], strides = [ 1, 2, 2, 1 ], 
                       padding = 'SAME' )

reshape = tf.reshape( pool2, [ BatchSize * 2, -1 ] )
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss( shape = [ dim, FinalLocalSize ], stddev = 0.04, w1 = 0.004 )
bias3 = tf.Variable( tf.constant( 0.1, shape = [ FinalLocalSize ] ) )
local3 = tf.nn.relu( tf.matmul( reshape, weight3 ) + bias3 )

print(reshape)