from skimage import io
import os, sys
import numpy as np


def read_image( file_path ):
    img_list = os.listdir( file_path )
    print( img_list[100] )
    # img = np.zeros([len[(img_list)])
    tempimg = io.imread(os.path.join(file_path,img_list[0]))
    h, w, c = tempimg.shape
    img = np.zeros([len(img_list),h,w,c])
    for i in range(len(img_list)) :
        tempimg = io.imread(os.path.join(file_path,img_list[i]))
        img[i] = tempimg
    # tempimg = io.imread(os.path.join(file_path,img_list[133]))
    # print(tempimg[111,33])
    # print(img[133,111,33])
    return img

# read_image("G:\DML\数据库\VIPeRa\\all")