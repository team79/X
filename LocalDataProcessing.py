# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

from six.moves import urllib
import tarfile

# import tflearn
# from tflearn.data_utils import shuffle, to_categorical
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.estimator import regression
# from tflearn.data_preprocessing import ImagePreprocessing
# from tflearn.data_augmentation import ImageAugmentation

from datetime import datetime
import math
import time
from tensorflow.python.lib.io import file_io
import os
import sys
import numpy as np
import pickle
import argparse
import scipy
from skimage import io
import random
import collections # 原生的collections库
slim = tf.contrib.slim # 使用方便的contrib.slim库来辅助创建ResNet
#import read_image_in_pai

BatchSize = 128
TrainLen = 12936
ImageHeight = 128
ImageWidth = 64
ImageChannel = 3
FinalLocalSize = 2048
DataBlockNum = 6
TrainInit = False
ImageInit = False

def read_image( filepath ):
    img_obj = file_io.read_file_to_string(filepath)
    file_io.write_string_to_file("temp.jpg", img_obj)
    img = scipy.ndimage.imread("temp.jpg", mode="RGB")
    return img

def read_image_in_pai(FLAGS):
    img = np.zeros([TrainLen,ImageHeight, ImageWidth,3])
    label = np.zeros([TrainLen])
    dirname = os.path.join(FLAGS.buckets, "")
    files = tf.gfile.ListDirectory(dirname) 
    for i in range(len(files)) :
        if i % 1000 == 0:
            print("read the " + str(i+1) + "th image")
        imagepath = os.path.join(FLAGS.buckets, files[i])
        img[i] = read_image(imagepath)
        label[i] = int(files[i][0:4])
    return img, label


parser = argparse.ArgumentParser()
#获得buckets路径
parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
#获得checkpoint路径
parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
FLAGS, _ = parser.parse_known_args()

img, label = read_image_in_pai(FLAGS)
print(img[0,0,0,:])
print(label[0])
for i in range(DataBlockNum):
    dirname = os.path.join(FLAGS.checkpointDir, "Mk1501Image" + str(i+1) + ".txt")
    C = img[int((TrainLen/DataBlockNum*i)):int((TrainLen/DataBlockNum*(i+1))),:,:,:]
    print(C[0,0,0,:])
    C = C.tostring()
    print(len(C))
    file_io.write_string_to_file(dirname, C )
for i in range(DataBlockNum):
    dirname = os.path.join(FLAGS.checkpointDir, "Mk1501Label" + str(i+1) + ".txt")
    C = label[int((TrainLen/DataBlockNum*i)):int((TrainLen/DataBlockNum*(i+1)))]
    print(C[0])
    C = C.tostring()
    print(len(C))
    file_io.write_string_to_file(dirname, C )