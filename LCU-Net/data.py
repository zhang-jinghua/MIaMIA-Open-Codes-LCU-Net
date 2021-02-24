# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:59:47 2019

@author: zjh
"""
import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import skimage.io as io



def adjustData(original,mask):
    original = original/255
    mask = mask/255
    mask[mask > 0.5] = 1
    mask[mask < 0.5] =0
    return(original,mask)



def trainGenerator(batch_size,train_path,original_dir,mask_dir,aug_dict,target_size,image_color_mode = "grayscale",aug_image_save_dir=None,aug_mask_save_dir=None,original_aug_prefix="image",mask_aug_prefix="mask",seed=1):
    original_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    original_generator = original_datagen.flow_from_directory(
            train_path,
            classes = [original_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_image_save_dir,
            save_prefix = original_aug_prefix,
            seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
            train_path,
            classes = [mask_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_mask_save_dir,
            save_prefix = mask_aug_prefix,
            seed = seed)

    train_generator = zip(original_generator,mask_generator)



    for (original,mask) in train_generator:
        original,mask = adjustData(original,mask)
        yield (original,mask)

def validationGenerator(batch_size,train_path,original_dir,mask_dir,aug_dict,target_size,image_color_mode = "grayscale",aug_image_save_dir=None,aug_mask_save_dir=None,original_aug_prefix="image",mask_aug_prefix="mask",seed=1):
    original_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    original_generator = original_datagen.flow_from_directory(
            train_path,
            classes = [original_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_image_save_dir,
            save_prefix = original_aug_prefix,
            seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
            train_path,
            classes = [mask_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_mask_save_dir,
            save_prefix = mask_aug_prefix,
            seed = seed)



    train_generator = zip(original_generator,mask_generator)



    for (original,mask) in train_generator:
        original,mask = adjustData(original,mask)
        yield (original,mask)


def testGenerator(test_path, num_image, target_size):
    for i in range(num_image):
        img = cv.imread(test_path + "/" + str(i) + ".png", cv.IMREAD_GRAYSCALE)
        img = img / 255
        img = cv.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img


#


def saveResult(save_path, result, flag_multi_class=False, num_class=2):
    for i, item in enumerate(result):
        img = item[:, :, 0]
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)     #使用该函数保存时不用做归一化
        cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)  # 使用opencvimwrite保存图片时，需做归一化，将原先归一化至0-1的图片复原出来，否则结果全黑
        cv.imwrite(save_path + "/" + str(i) + "_predict.png", img)
