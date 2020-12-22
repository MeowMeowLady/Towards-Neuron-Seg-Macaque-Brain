# -*- coding: utf-8 -*-
"""
Created on 19-4-24 下午10:01
IDE PyCharm 

@author: Meng Dong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import pprint
import time
import logging
import numpy as np
import timeit

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

lib_path = os.path.join(this_dir, 'lib')
add_path(lib_path)


import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn

from models.CrowdCounter import CrowdCounter
from utils.utils import poly_scheduler, AverageMeter, create_logger
from config import config
from config import update_config
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


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
        'test_global_steps': 0,
    }

    rand_seed = config.TRAIN.SEED
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)
    gpus = list(config.GPUS)
    if len(gpus) == 1:
        torch.cuda.set_device(config.GPUS[0])


    # prepare data
    train_set = macaque_h5(config.DATA.DATA_PATH, config.DATA.TRAIN_SET,'train')
    train_loader = DataLoader(train_set, batch_size=config.TRAIN.BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True)

    val_set = macaque_h5(config.DATA.DATA_PATH, config.DATA.TEST_SET, 'valid')
    val_loader = DataLoader(val_set, batch_size=config.VAL.BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    net = CrowdCounter(config)

    # multi-gpu training

    net = nn.DataParallel(net, device_ids=[0,]).cuda()

    epoch_iters = np.int(train_set.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                  filter(lambda p: p.requires_grad,
                                         net.parameters()),
                                  'lr': config.TRAIN.LR}],
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')


    lowest_loss = np.inf
    last_epoch = 0
    start = timeit.default_timer()
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            lowest_loss = checkpoint['lowest_loss']
            last_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # training
        train(config, train_loader, net, optimizer,
            epoch, config.TRAIN.END_EPOCH, num_iters, epoch_iters, writer_dict)

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch + 1,
            'lowest_loss': lowest_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

        #save model
        torch.save(net.module.state_dict(),
                   os.path.join(final_output_dir, 'epoch{}.pth'.format(epoch)))


        # validation
        if config.TRAIN.METHOD == 'ours':
            valid_loss, valid_density_loss, valid_class_loss = valid(config, val_loader, net, writer_dict)
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                torch.save(net.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
            msg = 'Validation Loss: {:.6f}, Density_Loss: {:.6f}, Class_Loss: {:.6f}, Lowest_Loss: {: .6f}'.format(
                valid_loss, valid_density_loss, valid_class_loss, lowest_loss)
            logging.info(msg)
        elif config.TRAIN.METHOD == 'baseline':
            valid_loss, valid_voronoi_loss, valid_cluster_loss = valid(config, val_loader, net, writer_dict)
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                torch.save(net.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
            msg = 'Validation Loss: {:.6f}, Voronoi_Loss: {:.6f}, Cluster_Loss: {:.6f}, Lowest_Loss: {: .6f}'.format(
                valid_loss, valid_voronoi_loss, valid_cluster_loss, lowest_loss)
            logging.info(msg)

        # test
        #prec, rec, mdice = test(config, net, epoch, writer_dict)
        #msg = 'Test Precision: {:.6f}, Recall: {:.6f}, mDice: {:.6f}'.format(prec, rec, mdice)
        #logging.info(msg)

    # save model
    torch.save(net.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))
    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %f' % ((end - start) / 3600.))
    logger.info('Done')


def train(config, train_loader, net, optimizer, epoch, num_epoch, num_iters, epoch_iters, writer_dict):
    net.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    if config.TRAIN.METHOD == 'ours':
        ave_density_loss = AverageMeter()
        ave_class_loss = AverageMeter()

        tic = time.time()
        cur_iters = epoch * epoch_iters
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']

        for i_iter, batch in enumerate(train_loader, 0):

            img, gt_den, gt_cls, gt_vor, gt_clu, fname = batch
            img = img.float().cuda()
            gt_den = gt_den.float().cuda()
            gt_cls = gt_cls.float().cuda()
            gt_vor = gt_vor.cuda()
            gt_clu = gt_clu.cuda()

            gt_dic = {'gt_den': gt_den, 'gt_cls': gt_cls, 'gt_vor': gt_vor, 'gt_clu': gt_clu}
            optimizer.zero_grad()
            pred = net(img, gt_dic)

            loss_density, loss_class = net.module.loss
            loss = config.TRAIN.LOSS_WEIGHT[0] * loss_density + config.TRAIN.LOSS_WEIGHT[1] * loss_class

            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss
            ave_loss.update(loss.item())
            ave_density_loss.update(loss_density.item())
            ave_class_loss.update(loss_class.item())

            if i_iter % config.TRAIN.PRINT_FREQ == 0:

                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                      'lr: {:.6f}, Loss: {:.6f}, Density_Loss: {:.6}, Class_Loss: {:.6f}'.format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), optimizer.param_groups[0]['lr'], ave_loss.average(),
                      ave_density_loss.average(), ave_class_loss.average())

                logging.info(msg)
            poly_scheduler(optimizer, num_iters, cur_iters+i_iter, config)

    elif config.TRAIN.METHOD == 'baseline':
        ave_voronoi_loss = AverageMeter()
        ave_cluster_loss = AverageMeter()

        tic = time.time()
        cur_iters = epoch * epoch_iters
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']

        for i_iter, batch in enumerate(train_loader, 0):

            img, gt_den, gt_cls, gt_vor, gt_clu, fname = batch
            img = img.float().cuda()
            gt_den = gt_den.float().cuda()
            gt_cls = gt_cls.float().cuda()
            gt_vor = gt_vor.cuda()
            gt_clu = gt_clu.cuda()

            gt_dic = {'gt_den': gt_den, 'gt_cls': gt_cls, 'gt_vor': gt_vor, 'gt_clu': gt_clu}
            optimizer.zero_grad()
            pred = net(img, gt_dic)

            loss_voronoi, loss_cluster = net.module.loss
            loss = config.TRAIN.LOSS_WEIGHT[2] * loss_voronoi + config.TRAIN.LOSS_WEIGHT[3] * loss_cluster
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss
            ave_loss.update(loss.item())
            ave_voronoi_loss.update(loss_voronoi.item())
            ave_cluster_loss.update(loss_cluster.item())

            if i_iter % config.TRAIN.PRINT_FREQ == 0:
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                      'lr: {:.6f}, Loss: {:.6f}, Voronoi_Loss: {:.6}, Cluster_Loss: {:.6f}'.format(
                       epoch, num_epoch, i_iter, epoch_iters,
                       batch_time.average(), optimizer.param_groups[0]['lr'], ave_loss.average(),
                       ave_voronoi_loss.average(), ave_cluster_loss.average())
                logging.info(msg)

            poly_scheduler(optimizer, num_iters, cur_iters+i_iter, config)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def valid(config, val_loader, net, writer_dict):
    net.eval()
    ave_loss = AverageMeter()
    if config.TRAIN.METHOD == 'ours':
        ave_class_loss = AverageMeter()
        ave_density_loss = AverageMeter()
        #mae = AverageMeter()
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                img, gt_den, gt_cls, gt_vor, gt_clu, fname = batch
                img = img.float().cuda()
                gt_den = gt_den.float().cuda()
                gt_cls = gt_cls.float().cuda()
                gt_vor = gt_vor.cuda()
                gt_clu = gt_clu.cuda()
                gt_dic = {'gt_den': gt_den, 'gt_cls': gt_cls, 'gt_vor': gt_vor, 'gt_clu': gt_clu}
                pred = net(img, gt_dic)

                loss_density, loss_class = net.module.loss
                loss = config.TRAIN.LOSS_WEIGHT[0]*loss_density + config.TRAIN.LOSS_WEIGHT[1]*loss_class
                ave_loss.update(loss.item())
                ave_class_loss.update(loss_class.item())
                ave_density_loss.update(loss_density.item())

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
        writer.add_scalar('valid_density_loss', ave_density_loss.average(), global_steps)
        #writer.add_scalar('valid_mae', mae.average(), global_steps)
        writer.add_scalar('valid_class_loss', ave_class_loss.average(), global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        return ave_loss.average(), ave_density_loss.average(), ave_class_loss.average()
    elif config.TRAIN.METHOD == 'baseline':
        ave_voronoi_loss = AverageMeter()
        ave_cluster_loss = AverageMeter()
        # mae = AverageMeter()
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                img, gt_den, gt_cls, gt_vor, gt_clu, fname = batch
                img = img.float().cuda()
                gt_den = gt_den.float().cuda()
                gt_cls = gt_cls.float().cuda()
                gt_vor = gt_vor.cuda()
                gt_clu = gt_clu.cuda()

                gt_dic = {'gt_den': gt_den, 'gt_cls': gt_cls, 'gt_vor': gt_vor, 'gt_clu': gt_clu}
                pred = net(img, gt_dic)

                loss_voronoi, loss_cluster = net.module.loss
                loss = config.TRAIN.LOSS_WEIGHT[2] * loss_voronoi + config.TRAIN.LOSS_WEIGHT[3] * loss_cluster
                ave_loss.update(loss.item())
                ave_voronoi_loss.update(loss_voronoi.item())
                ave_cluster_loss.update(loss_cluster.item())

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
        writer.add_scalar('valid_voronoi_loss', ave_voronoi_loss.average(), global_steps)
        writer.add_scalar('valid_cluster_loss', ave_cluster_loss.average(), global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        return ave_loss.average(), ave_voronoi_loss.average(), ave_cluster_loss.average()



if __name__ == '__main__':
    main()








