
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

TrainInit = False
ImageInit = False


parser = argparse.ArgumentParser()
#获得buckets路径
parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
#获得checkpoint路径
parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
FLAGS, _ = parser.parse_known_args()

dirname = os.path.join(FLAGS.buckets, "Mk1501Image.txt")
c = np.fromstring(file_io.read_file_to_string(dirname)).reshape([TrainLen,ImageHeight, ImageWidth,3])
print(c.shape)
print(c[0,0,0,:])
dirname = os.path.join(FLAGS.buckets, "Mk1501Label.txt")
c = np.fromstring(file_io.read_file_to_string(dirname)).reshape([TrainLen,ImageHeight, ImageWidth,3])
print(c.shape)
print(c[0])
