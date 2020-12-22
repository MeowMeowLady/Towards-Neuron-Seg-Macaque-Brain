
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = './exp'
_C.LOG_DIR = './exp/log'
_C.GPUS = (0, 1, 2, 3)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.IN_CHANNELS = 1

_C.LOSS = CN()
_C.LOSS.CLASS_BALANCE = False

# DATASET related params
_C.DATA = CN()
_C.DATA.DATASET = 'macaque'
_C.DATA.SUFFIX = '.h5'
_C.DATA.NUM_CLASSES = 1
_C.DATA.DATA_PATH = ''
_C.DATA.LABEL_FACTOR = 1
_C.DATA.LOG_PARA = 100.
_C.DATA.MEAN = [548.27, 548.27]
_C.DATA.VAR = [332.33, 332.33]
_C.DATA.TRAIN_SET = ''
_C.DATA.TEST_SET = ''
_C.DATA.PEAK_ALPHA = 0.5

# training
_C.TRAIN = CN()
_C.TRAIN.IMAGE_SIZE = [20, 128, 128]  # slices * height * width

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01

_C.TRAIN.LOSS_WEIGHT = [1., 1., 1., 1.] # loss_density, loss_class, loss_voronoi, loss_cluster

_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.SEED = 3035
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1
_C.TRAIN.POWER = 0.9
_C.TRAIN.METHOD = 'baseline'


_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 10

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 2
_C.TRAIN.BATCH_SIZE = _C.TRAIN.BATCH_SIZE_PER_GPU * len(_C.GPUS)
_C.TRAIN.SHUFFLE = True


# validation
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 1


# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [20, 128, 128]  # slices * width * height

_C.TEST.BATCH_SIZE_PER_GPU = 1
_C.TEST.MODEL_FILE = ''
_C.TEST.PRM_ON = False
_C.TEST.FINAL_OUTPUT_PATH = ''
_C.TEST.PRM_OUTPUT_PATH = ''
_C.TEST.DATA_PATH = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

