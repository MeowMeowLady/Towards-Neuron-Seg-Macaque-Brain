# -*- coding: utf-8 -*-
"""
Created on 19-7-31 下午6:02
IDE PyCharm 

@author: Meng Dong
this script is used to split training and validation sets by generating list-txt.
"""

from os.path import join as opj
import random
from glob import glob
import os

txt_save_path = ''

total_list = os.listdir('/')
random.shuffle(total_list)
random.shuffle(total_list)

train_list = open(opj(txt_save_path, 'train.txt'), 'w+')
valid_list = open(opj(txt_save_path, 'valid.txt'), 'w+')

for i in range(3500):
    train_list.write('{}\n'.format(total_list[i][:-3]))

for i in range(1000):
    valid_list.write('{}\n'.format(total_list[i+3500][:-3]))

train_list.close()
valid_list.close()


