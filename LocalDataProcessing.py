# -*- coding: utf-8 -*-
# from __future__ import division, print_function, absolute_import

import tensorflow as tf

# from six.moves import urllib
# import tarfile

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
TrainLen = 632
ImageHeight = 128
ImageWidth = 48
ImageChannel = 3
FinalLocalSize = 2048
DataBlockNum = 1
TrainInit = False
ImageInit = False

# def read_image( filepath ):
#     img_obj = file_io.read_file_to_string(filepath)
#     file_io.write_string_to_file("temp.jpg", img_obj)
#     img = scipy.ndimage.imread("temp.jpg", mode="RGB")
#     return img

def read_image_in_pai():
    GalleryImg = np.zeros([TrainLen,ImageHeight, ImageWidth,3])
    ProbeImg = np.zeros([TrainLen,ImageHeight, ImageWidth,3])
    dirname = "G:\DML\数据库\VIPeRa\\all"
    print(dirname)
    files = os.listdir(dirname) 
    #print(files[3368])
    print(len(files))
    cnt = 0
    for i in range(len(files)) :
        if files[i][-4:-1] != ".bm":
            continueWWWWWW
        if i % 100 == 0:
            print("read the " + str(i+1) + "th image")
        imagepath = os.path.join(dirname, files[i])
        if files[i][6] == '1':
            ProbeImg[int(files[i][0:4])-1] = scipy.ndimage.imread(imagepath, mode="RGB")
        else:
            GalleryImg[int(files[i][0:4])-1] = scipy.ndimage.imread(imagepath, mode="RGB")
    return GalleryImg, ProbeImg


# parser = argparse.ArgumentParser()
# #获得buckets路径
# parser.add_argument('--buckets', type=str, default='',
#                         help='input data path')
# #获得checkpoint路径
# parser.add_argument('--checkpointDir', type=str, default='',
#                         help='output model path')
# FLAGS, _ = parser.parse_known_args()

GalleryImg, ProbeImg = read_image_in_pai()
for i in range(DataBlockNum):
    dirname = os.path.join("", "ViperProbeImage" + str(i+1) + ".txt")
    C = ProbeImg[int((TrainLen/DataBlockNum*i)):int((TrainLen/DataBlockNum*(i+1))),:,:,:]
    print(C[0,0,0,:])
    C = C.tostring()
    print(len(C))
    file_io.write_string_to_file(dirname, C )
for i in range(DataBlockNum):
    dirname = os.path.join("", "ViperGalleryImage" + str(i+1) + ".txt")
    C = GalleryImg[int((TrainLen/DataBlockNum*i)):int((TrainLen/DataBlockNum*(i+1)))]
    print(C[0])
    C = C.tostring()
    print(len(C))
    file_io.write_string_to_file(dirname, C )