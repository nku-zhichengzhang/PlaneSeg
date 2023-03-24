"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import sys
import cv2
import copy
import glob

from models.model import *
from models.refinement_net import RefineModel
from models.modules import *
from datasets.plane_stereo_dataset import PlaneDataset
from datasets.inference_dataset import InferenceDataset
from datasets.nyu_dataset import NYUDataset
from utils import *
from visualize_utils import *
from evaluate_utils import *
from plane_utils import *
from options import parse_args
from config import InferenceConfig


# 39 58 99 605 615

def evaluate(options):
    config = InferenceConfig(options)
    config.FITTING_TYPE = options.numAnchorPlanes
    dataset = PlaneDataset(options, config, split='test', random=False, load_semantics=False)
    print('the number of images', len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    epoch_losses = []
    data_iterator = tqdm(dataloader, total=len(dataset))
    path = '../scannetDepth/val/'

    for sampleIndex, sample in enumerate(data_iterator):
        depthID = sampleIndex - 1
        gt_depth = sample[8].numpy()[0][80:560, :]
        gt_depth = cv2.resize(gt_depth, (256, 192))
        gt_depth = np.expand_dims(gt_depth, 2)
        gt_depth = gt_depth.astype(np.float32)
        np.save(path + 'depth_%d.npy' % depthID, gt_depth)

        #
        # for indexOffset in [0, ]:
        #     images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = \
        #         sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[
        #             indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[
        #             indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[
        #             indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == '':
        args.keyname = 'evaluate'
    else:
        args.keyname = args.dataset
        pass
    args.test_dir = 'test/' + args.keyname

    if args.testingIndex >= 0:
        args.debug = True
        pass
    if args.debug:
        args.test_dir += '_debug'
        args.printInfo = True
        pass

    ## Write html for visualization
    if False:
        if False:
            info_list = ['image_0', 'segmentation_0', 'segmentation_0_warping', 'depth_0', 'depth_0_warping']
            writeHTML(args.test_dir, info_list, numImages=100, convertToImage=False, filename='index', image_width=256)
            pass
        if False:
            info_list = ['image_0', 'segmentation_0', 'detection_0_planenet', 'detection_0_warping',
                         'detection_0_refine']
            writeHTML(args.test_dir, info_list, numImages=20, convertToImage=True, filename='comparison_segmentation')
            pass
        if False:
            info_list = ['image_0', 'segmentation_0', 'segmentation_0_manhattan_gt', 'segmentation_0_planenet',
                         'segmentation_0_warping']
            writeHTML(args.test_dir, info_list, numImages=30, convertToImage=False, filename='comparison_segmentation')
            pass
        exit(1)
        pass

    if not os.path.exists(args.test_dir):
        os.system("mkdir -p %s" % args.test_dir)
        pass

    if args.debug and args.dataset == '':
        os.system('rm ' + args.test_dir + '/*')
        pass

    evaluate(args)
