# -*- coding: utf-8 -*-
"""
Created on 19-4-24 下午9:22
IDE PyCharm 

@author: Meng Dong
"""

from torch.utils.data import Dataset
import os
import h5py as hf
import torch
from config import config
import numpy as np

class macaque_h5(Dataset):
    def __init__(self, data_path, list_path, mode):
        self.data_path = data_path

        #self.data_files = [filename for filename in os.listdir(self.data_path) \
        #                   if os.path.isfile(os.path.join(self.data_path, filename))]
        self.data_files = [os.path.join(data_path, item.rstrip()+config.DATA.SUFFIX) for item in open(list_path)]
        self.num_samples = len(self.data_files)
        self.mode = mode
        self.gt_factor = config.DATA.LOG_PARA
        self.peak_alpha = config.DATA.PEAK_ALPHA


    def __getitem__(self, index):
        fname = self.data_files[index]
        if self.mode=='train' or self.mode == 'valid':
            input_size = config.TRAIN.IMAGE_SIZE
        else:
            input_size = config.TEST.IMAGE_SIZE

        img, gt_den, gt_cls, gt_vor, gt_clu, fname = self.read_image_and_gt(fname, input_size)

        img = self.img_transform(img)

        gt_den, gt_cls, gt_vor, gt_clu = self.gt_transform(gt_den, gt_cls, gt_vor, gt_clu)

        return img, gt_den, gt_cls, gt_vor, gt_clu, fname

    def __len__(self):
        return self.num_samples

    '''
    def read_image_and_gt(self, fname, input_size):
        h5f = hf.File(os.path.join(self.data_path, fname), 'r')
        img = h5f['image'].value#.T # hdf5 created by matlab
        den = h5f['density'].value#.T
        mask = h5f['mask'].value
        weight = h5f['weight'].value
        assert (list(img.shape) == input_size) & (img.shape == den.shape) & (img.shape == mask.shape) & (img.shape == weight.shape),\
            'the shape of image or density map does not match the input size'
        img = img[np.newaxis, :, :, :]
        den = den[np.newaxis, :, :, :]
        return img, den, mask, weight, fname
    '''

    def read_image_and_gt(self, fname, input_size):
        h5f = hf.File(os.path.join(self.data_path, fname), 'r')
        img = h5f['image'][()]
        den = h5f['density'][()]
        cls = h5f['prob'][()]
        vor = h5f['voronoi'][()]
        clu = h5f['cluster'][()]
        if len(img.shape) == 3:
            assert (list(img.shape) == input_size) & (img.shape == den.shape) & (img.shape == cls.shape),\
                'the shape of image or density map does not match the input size'
            img = img[np.newaxis, :]
        elif len(img.shape) == 4:
            assert (list((img.shape[0], img.shape[2], img.shape[3])) == input_size) & ((img.shape[0], img.shape[2], img.shape[3]) == den.shape) \
                   & ((img.shape[0], img.shape[2], img.shape[3]) == cls.shape), \
                'the shape of image or density map does not match the input size'
            img = np.transpose(img, (1, 0, 2, 3))

        den = den[np.newaxis, :]
        cls = cls[np.newaxis, :] # ch, s, h, w

        # encode voronoi and cluster labels
        new_label = np.ones((vor.shape[0], vor.shape[1], vor.shape[2]), dtype=np.uint8) * 2  # ignored
        new_label[vor[:, :, :, 0] == 255] = 0  # background
        new_label[vor[:, :, :, 1] == 255] = 1  # nuclei
        vor = new_label[:, :, :] # s, h, w

        new_label = np.ones((clu.shape[0], clu.shape[1], clu.shape[2]), dtype=np.uint8) * 2  # ignored
        new_label[clu[:, :, :, 0] == 255] = 0  # background
        new_label[clu[:, :, :, 1] == 255] = 1  # nuclei
        clu = new_label[:, :, :] # s, h, w

        return img, den, cls, vor, clu, fname

    def get_num_samples(self):
        return self.num_samples

    def img_transform(self, img):
        img = img.astype(np.float32)
        if img.shape[0] == 1:
            avg = config.DATA.MEAN
            var = config.DATA.VAR
            img = (img-avg[0])/var[0]
        elif img.shape[0] == 2:
            avg = config.DATA.MEAN
            var = config.DATA.VAR
            img[0,:] = (img[0,:] - avg[0]) / var[0]
            img[1, :] = (img[1, :] - avg[1]) / var[1]

        return img

    def gt_transform(self, den, cls, vor=None, clu=None):
        den = den.astype(np.float32)
        den = 1. / (1. + self.peak_alpha * den)
        den = den/np.max(den)
        den = np.concatenate((1. - den, den), axis=0)
        cls = cls.astype(np.float32)
        cls = np.concatenate((1. - cls, cls), axis=0)

        if vor is not None:
            vor = vor.astype(np.long)
            clu = clu.astype(np.long)
            return den, cls, vor, clu
        return den, cls

