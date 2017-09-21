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

image_holder = tf.placeholder( tf.float32, [BatchSize * 2, ImageHeight, ImageWidth, ImageChannel] )

weight1 = variable_with_weight_loss( shape = [7, 7, 3, 32 ], stddev = 5e-2, w1 = 0.0 )
kernel1 = tf.nn.conv2d( image_holder, weight1, [ 1, 3, 3, 1 ], padding = 'SAME' )
bias1 = tf.Variable( tf.constant( 0.0, shape = [32] ) )
global_conv1 = tf.nn.relu( tf.nn.bias_add( kernel1, bias1 ) )
# pool1 = tf.nn.max_pool( conv1, ksize = [ 1, 3, 3, 1 ], strides = [ 1, 2, 2, 1 ], padding = 'SAME' )
# norm1 = tf.nn.lrn( pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75 )

print("the first conv done!")

# tempheight = global_conv1.get_shape()[1].value
# tempheight = int(tempheight/4)
# part1_conv1 = global_conv1[:,0:tempheight,:,:]
# part2_conv1 = global_conv1[:,tempheight:tempheight*2,:,:]
# part3_conv1 = global_conv1[:,tempheight*2:tempheight*3,:,:]
# part4_conv1 = global_conv1[:,tempheight*3:tempheight*4,:,:]

#body feature
body_pool1 = tf.nn.max_pool( global_conv1, ksize = [ 1, 3, 3, 1 ], strides = [ 1, 3, 3, 1 ], padding = 'SAME' )
body_norm1 = tf.nn.lrn( body_pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75 )

body_weight2 = variable_with_weight_loss( shape = [5, 5, 32, 32 ], stddev = 5e-2, w1 = 0.0 )
body_kernel2 = tf.nn.conv2d( body_norm1, body_weight2, [ 1, 2, 2, 1 ], padding = 'SAME' )
body_bias2 = tf.Variable( tf.constant( 0.0, shape = [32] ) )
body_conv2 = tf.nn.relu( tf.nn.bias_add( body_kernel2, body_bias2 ) )

body_pool2 = tf.nn.max_pool( body_conv2, ksize = [ 1, 3, 3, 1 ], strides = [ 1, 2, 2, 1 ], padding = 'SAME' )
body_norm2 = tf.nn.lrn( body_pool2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75 )

print("the second conv done!")

body_reshape = tf.reshape( body_norm2, [ BatchSize * 2, -1 ] )
body_dim = body_reshape.get_shape()[1].value
body_weight3 = variable_with_weight_loss( shape = [ body_dim, 400 ], stddev = 0.04, w1 = 0.004 )
body_bias3 = tf.Variable( tf.constant( 0.1, shape = [ 400 ] ) )
body_local = tf.nn.relu( tf.matmul( body_reshape, body_weight3 ) + body_bias3 )

print("the third full connect done!")

# #part feature 1
# part1_weight2 = variable_with_weight_loss( shape = [3, 3, 32, 32 ], stddev = 5e-2, w1 = 0.0 )
# part1_kernel2 = tf.nn.conv2d( part1_conv1, part1_weight2, [ 1, 1, 1, 1 ], padding = 'SAME' )
# part1_bias2 = tf.Variable( tf.constant( 0.0, shape = [32] ) )
# part1_conv2 = tf.nn.relu( tf.nn.bias_add( part1_kernel2, part1_bias2 ) )

# part1_reshape = tf.reshape( part1_conv2, [ BatchSize * 2, -1 ] )
# part1_dim = part1_reshape.get_shape()[1].value
# part1_weight3 = variable_with_weight_loss( shape = [ part1_dim, 100 ], stddev = 0.04, w1 = 0.004 )
# part1_bias3 = tf.Variable( tf.constant( 0.1, shape = [ 100 ] ) )
# part1_local = tf.nn.relu( tf.matmul( part1_reshape, part1_weight3 ) + part1_bias3 )

# #part feature 2
# part2_weight2 = variable_with_weight_loss( shape = [3, 3, 32, 32 ], stddev = 5e-2, w1 = 0.0 )
# part2_kernel2 = tf.nn.conv2d( part2_conv1, part2_weight2, [ 1, 1, 1, 1 ], padding = 'SAME' )
# part2_bias2 = tf.Variable( tf.constant( 0.0, shape = [32] ) )
# part2_conv2 = tf.nn.relu( tf.nn.bias_add( part2_kernel2, part2_bias2 ) )

# part2_reshape = tf.reshape( part2_conv2, [ BatchSize * 2, -1 ] )
# part2_dim = part2_reshape.get_shape()[1].value
# part2_weight3 = variable_with_weight_loss( shape = [ part2_dim, 100 ], stddev = 0.04, w1 = 0.004 )
# part2_bias3 = tf.Variable( tf.constant( 0.1, shape = [ 100 ] ) )
# part2_local = tf.nn.relu( tf.matmul( part2_reshape, part2_weight3 ) + part2_bias3 )

# #part feature 3
# part3_weight2 = variable_with_weight_loss( shape = [3, 3, 32, 32 ], stddev = 5e-2, w1 = 0.0 )
# part3_kernel2 = tf.nn.conv2d( part3_conv1, part3_weight2, [ 1, 1, 1, 1 ], padding = 'SAME' )
# part3_bias2 = tf.Variable( tf.constant( 0.0, shape = [32] ) )
# part3_conv2 = tf.nn.relu( tf.nn.bias_add( part3_kernel2, part3_bias2 ) )

# part3_reshape = tf.reshape( part3_conv2, [ BatchSize * 2, -1 ] )
# part3_dim = part3_reshape.get_shape()[1].value
# part3_weight3 = variable_with_weight_loss( shape = [ part3_dim, 100 ], stddev = 0.04, w1 = 0.004 )
# part3_bias3 = tf.Variable( tf.constant( 0.1, shape = [ 100 ] ) )
# part3_local = tf.nn.relu( tf.matmul( part3_reshape, part3_weight3 ) + part3_bias3 )


# #part feature 4
# part4_weight2 = variable_with_weight_loss( shape = [3, 3, 32, 32 ], stddev = 5e-2, w1 = 0.0 )
# part4_kernel2 = tf.nn.conv2d( part4_conv1, part4_weight2, [ 1, 1, 1, 1 ], padding = 'SAME' )
# part4_bias2 = tf.Variable( tf.constant( 0.0, shape = [32] ) )
# part4_conv2 = tf.nn.relu( tf.nn.bias_add( part4_kernel2, part4_bias2 ) )

# part4_reshape = tf.reshape( part4_conv2, [ BatchSize * 2, -1 ] )
# part4_dim = part4_reshape.get_shape()[1].value
# part4_weight3 = variable_with_weight_loss( shape = [ part4_dim, 100 ], stddev = 0.04, w1 = 0.004 )
# part4_bias3 = tf.Variable( tf.constant( 0.1, shape = [ 100 ] ) )
# part4_local = tf.nn.relu( tf.matmul( part4_reshape, part4_weight3 ) + part4_bias3 )

# final_local = tf.concat([body_local, part1_local, part2_local, part3_local, part4_local], 1)
final_local = body_local


loss1 = tf.reduce_sum(tf.multiply(final_local[0]-final_local[0], final_local[0] - final_local[0]))
loss2 = tf.reduce_sum(tf.multiply(final_local[0]-final_local[0], final_local[0] - final_local[0]))

# image_dist = tf.reduce_sum(tf.multiply(final_local[0]-final_local[1], final_local[0] - final_local[1]))

for i in range( BatchSize ):
    print("the " + str(i) + "th person loss done!")
    for j in range( BatchSize ) :
        temp1 = tf.reduce_sum(tf.multiply(final_local[i]-final_local[i+BatchSize], final_local[i] - final_local[i+BatchSize]))
        temp2 = tf.reduce_sum(tf.multiply(final_local[i]-final_local[j+BatchSize], final_local[i] - final_local[j+BatchSize]))
        loss1 = tf.add( loss1, tf.maximum( -1.0, tf.subtract( temp1, temp2 ) ) )
        loss2 = tf.add( loss2, tf.maximum( 0.01, tf.reduce_sum(tf.multiply(final_local[i]-final_local[i+BatchSize], final_local[i] - final_local[i+BatchSize]) ) ) )

# loss = tf.div( tf.add( loss1, tf.multiply( loss2, 0.002 ) ), BatchSize * BatchSize * 1.0 )
loss = tf.add( loss1, tf.multiply( loss2, 0.002 ) )
train_op = tf.train.AdamOptimizer( 1e-3 ).minimize( loss )

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
        print("the " + str(i) + "th loss  :   " + str(t1) )
        tmpimg = np.zeros([BatchSize*2,ImageHeight,ImageWidth,ImageChannel])
        test_feature = np.zeros([TestLen*2,FinalLocalSize])
        for j in range( int(TestLen / BatchSize) ) :
            tmpimg[0 : BatchSize,:] = test_img[j*BatchSize:(j+1)*BatchSize,:]
            tmpimg[BatchSize : BatchSize * 2,:] = test_img[(j*BatchSize+TestLen):((j+1)*BatchSize+TestLen),:]
            t1 = sess.run(final_local, feed_dict={ image_holder : tmpimg })
            test_feature[j*BatchSize:(j+1)*BatchSize,:] = t1[0 : BatchSize,:]
            test_feature[(j*BatchSize+TestLen):((j+1)*BatchSize+TestLen),:] = t1[BatchSize : BatchSize * 2,:]
        tmpimg[0 : BatchSize,:] = test_img[(TestLen-BatchSize):(TestLen),:]
        tmpimg[BatchSize : BatchSize * 2,:] = test_img[(TestLen*2-BatchSize):(TestLen*2),:]
        t1 = sess.run(final_local, feed_dict={ image_holder : tmpimg })
        test_feature[(TestLen-BatchSize):(TestLen),:] = t1[0 : BatchSize,:]
        test_feature[(TestLen*2-BatchSize):(TestLen*2),:] = t1[BatchSize : BatchSize * 2,:]
        cmc = get_cmc( test_feature )
        print( [ cmc[i] for i in range( 0, 40, 5 )] )

# img = io.imread(FilePath + '\\0001001.bmp')
# print(FilePath)
# print(img.shape)
