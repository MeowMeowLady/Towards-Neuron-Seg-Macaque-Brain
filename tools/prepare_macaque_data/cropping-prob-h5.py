# -*- coding: utf-8 -*-
"""
Created on 19-7-30 下午5:27
IDE PyCharm 

@author: Meng Dong

this script is used to crop macaque image blocks into smaller ones
the cropped size is 20*128*128
"""

from skimage import io
from libtiff import TIFF
import os
from glob import glob
from xml.dom import minidom
import numpy as np
import h5py
from skimage.transform import resize
import mahotas

src_path = ''
dst_path = ''

norm_size = [20, 128, 128]

N = 0
M = 0
tif_list = glob(os.path.join(src_path, 'image', '*.tif'))
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
    # read image
    img = io.imread(tif)
    # read prob
    with h5py.File(os.path.join(src_path, 'ilastik-12-30', 'probability', name+'_Probabilities.h5'), 'r') as f:
        prob = f['exported_data'][:, 0, :, :, 0]

    # read voronoi
    vor = io.imread(os.path.join(src_path, 'labels_voronoi', name+'_label_vor.tif')) # s, h, w, ch
    # read cluster
    cluster = io.imread(os.path.join(src_path, 'labels_cluster', name+'_label_cluster.tif'))

    slices, ch, height, width = img.shape
    w = list(range(0, width-norm_size[2]+1, int(norm_size[2]/2)))
    h = list(range(0, height-norm_size[1]+1, int(norm_size[1]/2)))
    s = list(range(0, slices-norm_size[0]+1, norm_size[0]-9))


    # generate peak map
    bi_image = np.ones((slices, height, width), dtype=bool)
    bi_image = resize(bi_image, (int(slices * 2.5), height, width), anti_aliasing=True)

    seeds = np.zeros(bi_image.shape, dtype=int)
    for idx, p in enumerate(points):
        [px, py, pz] = p[:]
        px = np.minimum(width - 1, np.maximum(0, int(px)))
        py = np.minimum(height - 1, np.maximum(0, int(py)))
        pz = np.minimum(slices - 1, np.maximum(0, int(pz)))
        seeds[pz, py, px] = idx + 1
        bi_image[int(2.5 * pz), py, px] = 0

    # distance transform
    d_map = mahotas.distance(bi_image)
    d_map = resize(d_map, (slices, height, width), anti_aliasing=True)

    density = d_map

    for iss in s:
        for ih in h:
            for iw in w:
                crop_img = img[iss: iss+norm_size[0], :, ih: ih+norm_size[1], iw: iw+norm_size[2]]
                crop_prob = prob[iss: iss+norm_size[0], ih: ih+norm_size[1], iw: iw+norm_size[2]]
                crop_vor = vor[iss: iss+norm_size[0], ih: ih+norm_size[1], iw: iw+norm_size[2], :]
                crop_cluster = cluster[iss: iss+norm_size[0], ih: ih+norm_size[1], iw: iw+norm_size[2], :]
                keep = (points[:, 0] >= iw) & (points[:, 0] <= iw + norm_size[2]) & (points[:, 1] >= ih) & (
                        points[:, 1] <= ih + norm_size[1]) & (points[:, 2] >= iss) & (points[:, 2] <= iss + norm_size[0])
                if keep.sum() == 0:
                    continue
                points_kept = points[keep, :] - np.array([iw, ih, iss])

                N += 1

                crop_den = density[iss: iss+norm_size[0], ih: ih+norm_size[1], iw: iw+norm_size[2]]

                # save as hdf5 files
                hf = h5py.File(os.path.join(dst_path, 'h5data', '{:04d}.h5'.format(N)), 'w')
                hf.create_dataset('image', data=crop_img)
                hf.create_dataset('prob', data=crop_prob)
                hf.create_dataset('density', data=crop_den)
                hf.create_dataset('voronoi', data=crop_vor)
                hf.create_dataset('cluster', data=crop_cluster)

                hf.close()

                # for visualization, we save images as blow steps

                # save cropped image
                crop_save = TIFF.open(os.path.join(dst_path, 'image', '{:04d}.tif'.format(N)), mode='w')
                for c in range(norm_size[0]):
                    crop_save.write_image(crop_img[c, :], compression='lzw', write_rgb=True)
                crop_save.close()

                # save cropped prob
                crop_prob = (crop_prob * 255).astype(np.uint8)
                crop_save = TIFF.open(os.path.join(dst_path, 'prob', '{:04d}.tif'.format(N)), mode='w')
                for c in range(norm_size[0]):
                    crop_save.write_image(crop_prob[c, :], compression='lzw', write_rgb=False)
                crop_save.close()

                # save cropped density
                crop_den = (crop_den/np.max(crop_den)*255).astype(np.uint8)
                crop_save = TIFF.open(os.path.join(dst_path, 'density', '{:04d}.tif'.format(N)), mode='w')
                for c in range(norm_size[0]):
                    crop_save.write_image(crop_den[c, :], compression='lzw', write_rgb=False)
                crop_save.close()

                # save cropped voronoi
                io.imsave(os.path.join(dst_path, 'voronoi', '{:04d}.tif'.format(N)), crop_vor)

                # save cropped cluster
                io.imsave(os.path.join(dst_path, 'cluster', '{:04d}.tif'.format(N)), crop_cluster)
                print('[{}] {} done!'.format(name, N))





