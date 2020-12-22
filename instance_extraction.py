# -*- coding: utf-8 -*-
"""
Created on 19-12-5 下午10:43
IDE PyCharm 

@author: Meng Dong

instance segmentation with the detected peak points as seeds and predicted semantic map as input image.
using watershed to complete segmentation.

"""

from skimage import io
import mahotas
import numpy as np
from os.path import join as opj
from libtiff import TIFF
import scipy.ndimage.filters as filters
from skimage.measure import label

def random_rgb():
    r = np.random.rand()*255
    g = np.random.rand()*255
    b = np.random.rand()*255
    return np.array([r, g, b]).astype(np.uint8)

def instance_seg(pred_path, img_names):
    save_path = pred_path

    for img_name in img_names:

        # load test image
        pred = io.imread(opj(pred_path, img_name + '_pred.tif'))
        s, h, w = pred.shape

        # set seeds points for watershed
        detected = pred == filters.maximum_filter(pred, size=10)
        seeds = label(detected)

        # set background seeds according to the class probability
        seeds[pred<50] = np.max(seeds)+1

        # apply watershed to the predicted class probability map
        labeled = mahotas.cwatershed(np.max(pred) - pred, seeds)
        labeled = labeled.astype(np.uint16)

        # save grayscale image
        gray = labeled.copy()
        gray[labeled==np.max(seeds)] = 0
        io.imsave(opj(save_path, '{}_instance.tif'.format(img_name)), gray)

        # save color image
        color_img = np.zeros((3, s, h, w), dtype='uint8')
        idx_list = np.unique(labeled)
        for idx in idx_list:
            rgb = random_rgb()
            color_img[:, labeled == idx] = rgb[:, np.newaxis]
        color_img = color_img.transpose((1, 0, 2, 3))
        img_save = TIFF.open(opj(save_path, '{}_color.tif'.format(img_name)), mode='w')
        for c in range(color_img.shape[0]):
            img_save.write_image(color_img[c, :], compression='lzw', write_rgb=True)
        img_save.close()


if __name__ == '__main__':
    
    pred_path = ''
    instance_seg(pred_path)