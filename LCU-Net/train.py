from keras.models import *
from keras.layers import *
from keras.optimizers import *
import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import skimage.io as io

###################################################
##创建文件夹函数##
def makedir(path):

    folder = os.path.exists(path)

    if folder is True:
        print("文件夹已存在")
    else:
        os.makedirs(path)
        print("文件夹创建中")
        print("完成")
###################################################


###################################################
             ####   非常重要  ####

file_path = "C:/Users/zjh/Desktop/Unet_data/Random1"
i = m_class  # 微生物类别，共0-20类
n = train_num  # 第n次训练


image_size = (256, 256, 1)
data_size = (256, 256)


test_num = 10  # 测试图片数量
BATCH_SIZE = 2

Aug_originall = file_path + "/" + str(i) + "/aug/original_" + str(n)  #训练集数据扩充文件夹
Aug_GTM1 = file_path + "/" + str(i) + "/aug/GTM_" + str(n)
Aug_original2 = file_path + "/" + str(i) + "/aug/val_original_" + str(n) #验证集数据扩充文件夹
Aug_GTM2 = file_path + "/" + str(i) + "/aug/val_GTM_" + str(n)
makedir(Aug_originall)
makedir(Aug_GTM1)
makedir(Aug_original2)
makedir(Aug_GTM2)
train_path = file_path + "/" + str(i) + "/train"
test_path = file_path + "/" + str(i) + "/test"
val_path = file_path + "/" + str(i) + "/val"
##################################################











