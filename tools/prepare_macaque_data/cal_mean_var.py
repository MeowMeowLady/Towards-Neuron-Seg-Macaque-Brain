# -*- coding: utf-8 -*-
"""
Created on 19-7-31 下午5:43
IDE PyCharm 

@author: Meng Dong

this script is used to calculate the mean and variance value for gray-scale images.
"""

import numpy as np
from skimage import io
from os.path import join as opj
from glob import glob

tif_list = glob(opj('./data/image', '*.tif'))
N = 0
val = 0
mean = 481.28
for tif in tif_list:
    img = io.imread(tif)
    img = img[:,1,:,:]
    img = img.astype(np.float)
    val += np.sum(img)
    #val += np.sum(np.square(img-mean))
    N += img.size

print(val/N)

