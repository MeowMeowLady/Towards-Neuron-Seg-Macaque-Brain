# -*- coding: utf-8 -*-
"""
Created on 19-12-26 下午9:32
IDE PyCharm 

@author: Meng Dong
"""

import numpy as np
from skimage import io
from tools.evaluation.mask_iou import mask_iou_fast
import os
import nibabel as nib

def calc_instance_segmentation_voc_prec_rec(
        pred_mask_path, gt_mask_path, img_names, iou_thresh):

    n_pos = 0
    match = []
    save = False
    dice = []
    if save:
        fid = open(os.path.join('.txt'), 'w+')
    for img_name in img_names:
        match_single = []
        dice_single = []
        print('img {}'.format(img_name))
        # the pred_mask and gt_mask are uint16 3d array, each instance has an id
        pred_mask = io.imread(os.path.join(pred_mask_path, img_name+'_instance.tif'))
        gt = nib.load(os.path.join(gt_mask_path, img_name+'_label.nii.gz'))
        gt_mask = gt.get_fdata().transpose((2, 1, 0)).astype('uint16')

        s, h, w = pred_mask.shape[:]

        # get all the ids of instance and remove background id 0
        pred_mask_ids = np.unique(pred_mask).tolist()
        pred_mask_ids.remove(0)
        ##if len(pred_mask_ids) != len(pred_score):
        ##    print('error: the number of mask ids do not equal to score length!')
        gt_mask_ids = np.unique(gt_mask).tolist()
        gt_mask_ids.remove(0)
        # update n_pos
        n_pos += len(gt_mask_ids)
        # get pred masks for each instance and save as bool array
        pred_masks = np.zeros((len(pred_mask_ids), s, h, w), dtype=np.bool)
        for i, id in enumerate(pred_mask_ids):
            pred_masks[i, :] = pred_mask==id
        if len(pred_mask_ids)==0:
            continue
        if len(gt_mask_ids) == 0:
            match.extend([0,]*len(pred_mask_ids))
            match_single.extend([0, ] * len(pred_mask_ids))
            dice_single.extend([0, ]* len(pred_mask_ids))
        # get gt masks
        gt_masks = np.zeros((len(gt_mask_ids), s, h, w), dtype=np.bool)
        for i, gt_mask_id in enumerate(gt_mask_ids):
            gt_masks[i, :] = gt_mask==gt_mask_id

        #t1 = time()
        iou, tp_voxel, fp_voxel, fn_voxel = mask_iou_fast(pred_masks, gt_masks)
        gt_index = iou.argmax(axis=1)
        # set -1 if there is no matching ground truth
        gt_index[iou.max(axis=1) < iou_thresh] = -1
        #del iou

        selec = np.zeros(len(gt_mask_ids), dtype=bool)
        for mask_idx, gt_idx in enumerate(gt_index): # for each segmented instance mask
            if gt_idx >= 0:
                if not selec[gt_idx]:
                    match.append(1)
                    match_single.append(1)
                    # calculate dice
                    d = 2*tp_voxel[mask_idx, gt_idx] / (2*tp_voxel[mask_idx, gt_idx] + fn_voxel[mask_idx, gt_idx] + fp_voxel[mask_idx, gt_idx])
                    dice_single.append(d)
                    dice.append(d)
                else:
                    match.append(0)
                    match_single.append(0)
                    dice_single.append(0)
                    dice.append(0)

                selec[gt_idx] = True
            else:
                match.append(0)
                match_single.append(0)
                dice_single.append(0)
                dice.append(0)

        match_single = np.array(match_single, dtype=np.int8)

        tp_single = np.sum(match_single == 1)
        fp_single = np.sum(match_single == 0)

        # If an element of fp + tp is 0,
        # the prec is nan.
        prec_single = tp_single / (fp_single + tp_single)
        mdice_single = np.mean(dice_single)

        print('prec: {}, dice: {}'.format(prec_single, mdice_single))
        if save:
            fid.write('{}: {:.4f}\n'.format(img_name, prec_single))
    if save:
        fid.close()

    match = np.array(match, dtype=np.int8)

    tp = np.sum(match == 1)
    fp = np.sum(match == 0)

    # If an element of fp + tp is 0,
    # the prec is nan.
    prec = tp / (fp + tp)
    mdice = np.mean(dice)
    # If n_pos is 0, rec is None.
    if n_pos > 0:
        rec = tp / n_pos

    return prec, rec, mdice


if __name__ == "__main__":
    
    pred_mask_path = ''
    gt_mask_path = ''
    img_names = ['RM006-136-8-3371-12-223-0-0-64-0', 'RM006-141-19-9056-460-194-1024-16-0-64',
                 'RM007-191-6-11298-460-186-1536-0-128-0']
    iou = 0.3
    print(pred_mask_path)
    for i in range(0, 1):
        pred_mask_path_tmp = pred_mask_path #+ '{}'.format(i)
        prec, rec, mdice = calc_instance_segmentation_voc_prec_rec( pred_mask_path_tmp, gt_mask_path, img_names, iou_thresh=iou)
        print('iou_threshold: {}'.format(iou))
        print('epoch{} mprec: {}, mrec: {}, f1: {}, mdice: {}\n'.format(i, prec, rec, (2.*rec*prec)/(rec+prec), mdice))