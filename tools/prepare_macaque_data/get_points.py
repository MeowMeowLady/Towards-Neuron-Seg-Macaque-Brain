# -*- coding: utf-8 -*-
"""
Created on 20-7-9 下午8:39
IDE PyCharm 

@author: Meng Dong
"""

from skimage import io
from libtiff import TIFF
import os
from glob import glob
from xml.dom import minidom
import numpy as np

src_path = ''
dst_path = 'data/train/point'

norm_size = [20, 128, 128]

M = 0
tif_list = glob(os.path.join(src_path, 'image-single-channel', '*.tif'))
for tif in tif_list:
    name = tif.split('/')[-1][:-4]

    # read xml label file and saved in points
    xml = os.path.join(src_path, 'label', 'CellCounter_'+name+'.xml')
    dom_xml = minidom.parse(xml)
    root = dom_xml.documentElement
    markers = root.getElementsByTagName('Marker')
    points = np.empty((0, 3), dtype=int)
    for marker in markers:
        if np.float(marker.getElementsByTagName('MarkerZ')[0].firstChild.data) < 0:
            continue
        x = np.int(marker.getElementsByTagName('MarkerX')[0].firstChild.data)
        y = np.int(marker.getElementsByTagName('MarkerY')[0].firstChild.data)
        z = np.int((np.float(marker.getElementsByTagName('MarkerZ')[0].firstChild.data) - 1)/ 2.)
        points = np.concatenate((points, np.array([[x, y, z],])), axis=0)

    M += len(points)

    # save points label
    txt = open(os.path.join(dst_path, '{}.txt'.format(name)), 'w+')
    for p in points:
        txt.write('{} {} {}\n'.format(p[0], p[1], p[2]))
    txt.close()
    print('[{}] {} done!'.format(name, M))