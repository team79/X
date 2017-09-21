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

# read_image("G:\DML\数据库\VIPeRa\\all")