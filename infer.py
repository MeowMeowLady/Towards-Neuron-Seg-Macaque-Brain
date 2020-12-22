# -*- coding: utf-8 -*-
"""
Created on 19-4-26 下午4:56
IDE PyCharm 

@author: Meng Dong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import os
import numpy as np
from skimage import io
from libtiff import TIFF
from glob import glob
import pprint


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

lib_path = os.path.join(this_dir, 'lib')
add_path(lib_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as standard_transforms
from torch.utils.data import DataLoader


from tools import transforms as own_transforms
from prm.peak_response_mapping_3d import PeakResponseMapping_3d
from models.CrowdCounter import CrowdCounter
from utils.modelsummary import get_model_summary
from config import config
from config import update_config
from utils.utils import AverageMeter, create_logger
from datasets import macaque_h5

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()

    logger, final_output_dir = create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    gpus = list(config.GPUS)
    if len(gpus) == 1:
        torch.cuda.set_device(config.GPUS[0])

    CC = CrowdCounter(config).cuda()
    net = CC
    net = net.eval()
    for i in range(1):
        out_path = '/media/dongmeng/Data/Code/cell-count-pt/instance_segmentation_macaque_bithub_ours/exp/03101605/best'
        #
        model_file = '/media/dongmeng/Data/Code/cell-count-pt/instance_segmentation_macaque_bithub_ours/exp/03101605/best.pth'
            #
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        #if not os.path.exists(config.TEST.FINAL_OUTPUT_PATH):
        #    os.makedirs(config.TEST.FINAL_OUTPUT_PATH)


        if config.TEST.MODEL_FILE:
            model_state_file = config.TEST.MODEL_FILE
        else:
            #model_state_file = os.path.join(final_output_dir,
            #                                'final_state.pth')
            model_state_file = model_file
        logger.info('=> loading model from {}'.format(model_state_file))

        check_point = torch.load(model_state_file)
        net_dict = net.state_dict()
        pretrained_dict = {k[7:11]+k[18:]: v for k, v in check_point.items()}# if k[7:] in net_dict}
        for k, _ in pretrained_dict.items():
            logger.info(
                '=> loading {} from pretrained model'.format(k))
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)

        test_imgs = glob('/media/dongmeng/Hulk/dataset/cell-count-macaque/test/*.tif')
        for _, fpath in enumerate(test_imgs):
            img = io.imread(fpath)
            if len(img.shape) == 3:
                img = img[np.newaxis, :]
            elif len(img.shape) == 4:
                img = np.transpose(img, (1, 0, 2, 3))

            img = img_transform(config, img)
            img = img[np.newaxis, :]

            img = torch.from_numpy(img)
            img = img.float().cuda()

            # forward
            pred_map = net.test_forward(img)

            pred = pred_map.cpu().data.numpy()[0, 1, :, :, :]
            pred_img = (pred - np.min(pred)) / np.max(pred) * 255
            pred_img = pred_img.astype(np.uint8)
            io.imsave(os.path.join(out_path, fpath.split('/')[-1][:-4] + '_pred.tif'), pred_img)

            del pred_map


            print('img_name {} '.format(fpath.split('/')[-1][:-4]))



def img_transform(config, img):
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


if __name__ == '__main__':
    main()


