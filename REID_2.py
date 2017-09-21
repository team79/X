# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

from six.moves import urllib
import tarfile

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

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

BatchSize = 30
SampleLen = 632
TrainLen = 316
TestLen = 316
ImageHeight = 128
ImageWidth = 48
ImageChannel = 3
FinalLocalSize = 400

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

image_holder = tf.placeholder( tf.float32, [BatchSize, ImageHeight, ImageWidth, ImageChannel] )
label_holder = tf.placeholder( tf.int32, [BatchSize] )

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

reshape = tf.reshape( pool2, [ batch_size, -1 ] )
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss( shape = [ dim, SampleLen ], stddev = 0.04, w1 = 0.004 )
bias3 = tf.Variable( tf.constant( 0.1, shape = [ SampleLen ] ) )
local3 = tf.nn.relu( tf.matmul( reshape, weight3 ) + bias3 )

# weight4 = variable_with_weight_loss( shape = [ SampleLen, SampleLen ], stddev = 0.04, w1 = 0.004 )
# bias4 = tf.Variable( tf.constant( 0.1, shape = [ SampleLen ] ) )
# local4 = tf.nn.relu( tf.matmul( local3, weight4 ) + bias4 )

weight5 = variable_with_weight_loss( shape = [ SampleLen, TrainLen ], stddev = 1 / 192.0, w1 = 0.0 )
bias5 = tf.Variable( tf.constant( 0.0, shape = [ TrainLen ] ) )
logits = tf.add( tf.matmul( local3, weight5 ), bias5 )

def loss( logits, labels ):
    labels = tf.cast( labels, tf.int64 )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( 
            logits = logits, labels = labels, name = 'cross_entropy_per_example' )
    cross_entropy_mean = tf.reduce_mean( cross_entropy, name = 'cross_entropy' )
    tf.add_to_collection( 'losses', cross_entropy_mean )

    return tf.add_n( tf.get_collection( 'losses'), name = 'total_loss' )

loss = loss( logits, label_holder )
train_op = tf.train.AdamOptimizer( 1e-3 ).minimize( loss )
top_k_op = tf.nn.in_top_k( logits, label_holder, 1 )
person_probability = tf.nn.softmax( logits )
print("loss done!")

# #-----------------------------------------------------------------------------------------------------------------------------
# #------------------------------------------------- function define  ----------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------------------

def read_image( filepath ):
    img_obj = file_io.read_file_to_string(filepath)
    file_io.write_string_to_file("temp.jpg", img_obj)
    img = scipy.ndimage.imread("temp.jpg", mode="RGB")
    return img

def read_image_in_pai(FLAGS):
    img = np.zeros([1264,128,48,3])
    dirname = os.path.join(FLAGS.buckets, "")
    files = tf.gfile.ListDirectory(dirname) 
    for i in range(len(files)) :
        imagepath = os.path.join(FLAGS.buckets, files[i])
        img[i] = read_image(imagepath)
    # tempimg = io.imread(os.path.join(file_path,img_list[133]))
    # print(tempimg[111,33])
    # print(img[133,111,33])
    return img

def randimg( image, index ):
    id1 = random.sample(index,BatchSize)
    id2 = [ i + SampleLen for i in id1 ]
    id = id1 + id2 
    return image[id]

def get_cmc( image_feature ):
    dist = np.zeros([TestLen,TestLen])
    for i in range( TestLen ) :
        for j in range( TestLen ) : 
            dist[i,j] = sum( ( image_feature[i] - image_feature[j+TestLen] ) * ( image_feature[i] - image_feature[j+TestLen] ) )
    cmc = np.zeros([TestLen,TestLen])
    for i in range( TestLen ) :
        cnt = 0
        for j in range( TestLen ) :
            if dist[i,j] < dist[i,i] : 
                cnt += 1
        cmc[i,cnt] = 1.0
    cmc = sum( cmc )
    presum = 0
    for i in range( TestLen ) :
        presum += cmc[i]
        cmc[i] = presum
    cmc = cmc / TestLen 
    return cmc


# #-----------------------------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
#获得buckets路径
parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
#获得checkpoint路径
parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
FLAGS, _ = parser.parse_known_args()


print("----------------------------------------\n" * 2)
print("read image\n")
img = read_image_in_pai(FLAGS)
print("image size") 
print(img.shape)

print("----------------------------------------\n" * 2)
print("session initial")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_index = random.sample([ i for i in range(SampleLen)],TrainLen)
test_index = list( set( [ i for i in range(SampleLen) ] ) - set( train_index ) )

id11 = test_index
id22 = [ i + SampleLen for i in id11 ]
idd = id11 + id22
test_img = img[idd]

for i in range(10000):
    if i % 100 == 0:
        print("----------------------------------------\n" * 2)
        print(" the " + str(i) + "th :")
    tmpimg = randimg( img, train_index )
    if i % 100 == 0:
        print(" the " + str(i) + "th begin :")
    _, t1 = sess.run([ train_op, loss], feed_dict={ image_holder : tmpimg })
    #print("----------------------------------------\n" * 2)
    if i % 100 == 0:
        

# img = io.imread(FilePath + '\\0001001.bmp')
# print(FilePath)
# print(img.shape)
