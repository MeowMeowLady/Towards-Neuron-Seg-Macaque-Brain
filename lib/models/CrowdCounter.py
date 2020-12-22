# -*- coding: utf-8 -*-
"""
Created on 19-4-25 下午3:05
IDE PyCharm 

@author: Meng Dong
"""

import torch
import torch.nn as nn
import numpy as np
import lib.models as models
import torch.nn.functional as F

class CrowdCounter(nn.Module):
    def __init__(self, cfg):
        super(CrowdCounter, self).__init__()
        self.CCN = eval('models.' + cfg.MODEL.NAME + '.get_seg_model')(cfg)

        if len(cfg.GPUS) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=cfg.GPUS).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        self.loss_cross_entropy_fn = nn.CrossEntropyLoss(reduce=False).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.baseline_criterion = torch.nn.NLLLoss(ignore_index=2).cuda()
        self.method = cfg.TRAIN.METHOD

    @property
    def loss(self):
        if self.method == 'ours':
            return self.loss_cross_entropy_den, self.loss_cross_entropy_cls
        elif self.method == 'baseline':
            return self.loss_vor, self.loss_cluster


    def forward(self, img, gt_dic):
        gt_den = gt_dic['gt_den']
        gt_cls = gt_dic['gt_cls']
        gt_vor = gt_dic['gt_vor']
        gt_clu = gt_dic['gt_clu']

        pred_map = self.CCN(img)

        if self.method == 'ours':
            self.loss_cross_entropy_cls = torch.mean(torch.sum(-gt_cls*self.logsoftmax(pred_map), 1))
            self.loss_cross_entropy_den = torch.mean(torch.sum(-gt_den * self.logsoftmax(pred_map), 1))
            pred = self.softmax(pred_map)
        elif self.method == 'baseline':
            log_prob_maps = F.log_softmax(pred_map, dim=1)
            self.loss_vor = self.baseline_criterion(log_prob_maps, gt_vor)
            self.loss_cluster = self.baseline_criterion(log_prob_maps, gt_clu)
            pred = self.softmax(pred_map)
        else:
            #""To do""
            pred = None

        return pred

    def test_forward(self, img):
        #density_map, mask_map = self.CCN(img)
        #cls_pred = mask_map.argmax(1)

        pred_map = self.CCN(img)
        pred = self.softmax(pred_map)

        return pred #density_map, mask_map

