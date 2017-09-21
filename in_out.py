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

def read_image( filepath ):
    img_obj = file_io.read_file_to_string(filepath)
    file_io.write_string_to_file("temp.jpg", img_obj)
    img = scipy.ndimage.imread("temp.jpg", mode="RGB")
    return img

def write_file( filepath, s ):
    tf.gfile.FastGFile(filepath, 'wb').write(s)

def main(_):
    parser = argparse.ArgumentParser()
#获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
#获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()

    dirname = os.path.join(FLAGS.buckets, "")
    files = tf.gfile.ListDirectory(dirname) 
    imagepath = os.path.join(FLAGS.buckets, files[0])
    
    print(imagepath)
    img = read_image( imagepath )
    x = np.array(img)
    print( x.shape )

    # sess = tf.Session()
    # a = tf.placeholder( tf.float32, [ 2, 3 ] )
    # b = tf.constant(0.1,shape=[2,2])
    # c = tf.constant(0.1,shape=[1,3])
    # logits = tf.reduce_sum( tf.add( tf.matmul( b, a ), c ), 1 )
    # xxx = np.array([[1,1,1],[1,1,1]])
    # t = sess.run(logits,feed_dict = {a:xxx})
    # print(t)

    # xxx = os.path.join(FLAGS.buckets, "ttt.txt")
    # write_file( xxx, "000000000000000\n0000000000000000\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    #获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)