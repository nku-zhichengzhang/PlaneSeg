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
class PlaneRCNNDetector():
    def __init__(self, options, config, modelType, checkpoint_dir=''):
        self.options = options
        self.config = config
        self.modelType = modelType
        self.model = MaskRCNN_edge_fpn_resolution(config)
        self.model.cuda()
        self.model.eval()

        if modelType == 'basic':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/pair_' + options.anchorType
        elif modelType == 'pair':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/pair_' + options.anchorType
        elif modelType == 'refine':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/instance_' + options.anchorType
        elif modelType == 'refine_single':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/refinement_' + options.anchorType
        elif modelType == 'occlusion':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/plane_' + options.anchorType
        elif modelType == 'final':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/planercnn_' + options.anchorType
            pass

        if options.suffix != '':
            checkpoint_dir += '_' + options.suffix
            pass

        ## Indicates that the refinement network is trained separately        
        separate = modelType == 'refine'

        checkpoint_dir += '_ablation_edge_fpn_resolution'

        if not separate:
            if options.startEpoch >= 0:
                self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_' + str(options.startEpoch) + '.pth'))
            else:
                self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint.pth'))
                pass
            pass

        if 'refine' in modelType or 'final' in modelType:
            self.refine_model = RefineModel(options)

            self.refine_model.cuda()
            self.refine_model.eval()
            if not separate:
                state_dict = torch.load(checkpoint_dir + '/checkpoint_refine.pth')
                self.refine_model.load_state_dict(state_dict)
                pass
            else:
                self.model.load_state_dict(torch.load('checkpoint/pair_' + options.anchorType + '_pair/checkpoint.pth'))
                self.refine_model.load_state_dict(torch.load('checkpoint/instance_normal_refine_mask_softmax_valid/checkpoint_refine.pth'))
                pass
            pass

        return

    def detect(self, sample):

        input_pair = []
        detection_pair = []
        camera = sample[30][0].cuda()
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()
            p_fea, e_fea, f_fea = self.model.vis_ps_fea([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='inference_detection', use_nms=2, use_refinement=True)
            
        return p_fea, e_fea, f_fea


def draw_feature(x, suffix='L2', post = 'edge'):
    x=x.cpu()#.data.numpy()
    for i in range(0,x.shape[1],10):
        feature=x[0,i,:,:].view(x.shape[-2],x.shape[-1])
        feature=feature.cpu().detach().numpy()
        feature-=feature.min()
        feature/=feature.max()+1e-10
        feature=np.round(feature*255)

        dst_path = suffix+'_C'+str(i).zfill(3)+'_'+post+'.png'
        print(dst_path)
        cv2.imwrite(dst_path,feature)


def evaluate(options):
    config = InferenceConfig(options)
    config.FITTING_TYPE = options.numAnchorPlanes

    if options.dataset == '':
        dataset = PlaneDataset(options, config, split='test', random=False, load_semantics=False)
    elif options.dataset == 'occlusion':
        config_dataset = copy.deepcopy(config)
        config_dataset.OCCLUSION = False
        dataset = PlaneDataset(options, config_dataset, split='test', random=False, load_semantics=True)
    elif 'nyu' in options.dataset:
        dataset = NYUDataset(options, config, split='val', random=False)
    elif options.dataset == 'synthia':
        dataset = SynthiaDataset(options, config, split='val', random=False)
    elif options.dataset == 'kitti':
        camera = np.zeros(6)
        camera[0] = 9.842439e+02
        camera[1] = 9.808141e+02
        camera[2] = 6.900000e+02
        camera[3] = 2.331966e+02
        camera[4] = 1242
        camera[5] = 375
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/KITTI/scene_3/*.png'), camera=camera)
    elif options.dataset == '7scene':
        camera = np.zeros(6)
        camera[0] = 519
        camera[1] = 519
        camera[2] = 320
        camera[3] = 240
        camera[4] = 640
        camera[5] = 480
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/SevenScene/scene_3/*.png'), camera=camera)
    elif options.dataset == 'tanktemple':
        camera = np.zeros(6)
        camera[0] = 0.7
        camera[1] = 0.7
        camera[2] = 0.5
        camera[3] = 0.5
        camera[4] = 1
        camera[5] = 1
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/TankAndTemple/scene_4/*.jpg'), camera=camera)
    elif options.dataset == 'make3d':
        camera = np.zeros(6)
        camera[0] = 0.7
        camera[1] = 0.7
        camera[2] = 0.5
        camera[3] = 0.5
        camera[4] = 1
        camera[5] = 1
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/Make3D/*.jpg'), camera=camera)
    elif options.dataset == 'popup':
        camera = np.zeros(6)
        camera[0] = 0.7
        camera[1] = 0.7
        camera[2] = 0.5
        camera[3] = 0.5
        camera[4] = 1
        camera[5] = 1
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/PhotoPopup/*.jpg'), camera=camera)
    elif options.dataset == 'cross' or options.dataset == 'cross_2':
        image_list = ['test/cross_dataset/' + str(c) + '_image.png' for c in range(12)]
        cameras = []
        camera = np.zeros(6)
        camera[0] = 587
        camera[1] = 587
        camera[2] = 320
        camera[3] = 240
        camera[4] = 640
        camera[5] = 480
        for c in range(4):
            cameras.append(camera)
            continue
        camera_kitti = np.zeros(6)
        camera_kitti[0] = 9.842439e+02
        camera_kitti[1] = 9.808141e+02
        camera_kitti[2] = 6.900000e+02
        camera_kitti[3] = 2.331966e+02
        camera_kitti[4] = 1242.0
        camera_kitti[5] = 375.0
        for c in range(2):
            cameras.append(camera_kitti)
            continue
        camera_synthia = np.zeros(6)
        camera_synthia[0] = 133.185088
        camera_synthia[1] = 134.587036
        camera_synthia[2] = 160.000000
        camera_synthia[3] = 96.000000
        camera_synthia[4] = 320
        camera_synthia[5] = 192
        for c in range(2):
            cameras.append(camera_synthia)
            continue
        camera_tanktemple = np.zeros(6)
        camera_tanktemple[0] = 0.7
        camera_tanktemple[1] = 0.7
        camera_tanktemple[2] = 0.5
        camera_tanktemple[3] = 0.5
        camera_tanktemple[4] = 1
        camera_tanktemple[5] = 1
        for c in range(2):
            cameras.append(camera_tanktemple)
            continue
        for c in range(2):
            cameras.append(camera)
            continue
        dataset = InferenceDataset(options, config, image_list=image_list, camera=cameras)
    elif options.dataset == 'selected':
        image_list = glob.glob('test/selected_images/*_image_0.png')
        image_list = [filename for filename in image_list if '63_image' not in filename and '77_image' not in filename] + [filename for filename in image_list if '63_image' in filename or '77_image' in filename]
        camera = np.zeros(6)
        camera[0] = 587
        camera[1] = 587
        camera[2] = 320
        camera[3] = 240
        camera[4] = 640
        camera[5] = 480
        dataset = InferenceDataset(options, config, image_list=image_list, camera=camera)
    elif options.dataset == 'comparison':
        image_list = ['test/comparison/' + str(index) + '_image_0.png' for index in [65, 11, 24]]
        camera = np.zeros(6)
        camera[0] = 587
        camera[1] = 587
        camera[2] = 320
        camera[3] = 240
        camera[4] = 640
        camera[5] = 480
        dataset = InferenceDataset(options, config, image_list=image_list, camera=camera)
    elif 'inference' in options.dataset:
        image_list = glob.glob(options.customDataFolder + '/*.png') + glob.glob(options.customDataFolder + '/*.jpg')
        if os.path.exists(options.customDataFolder + '/camera.txt'):
            camera = np.zeros(6)
            with open(options.customDataFolder + '/camera.txt', 'r') as f:
                for line in f:
                    values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
                    for c in range(6):
                        camera[c] = values[c]
                        continue
                    break
                pass
        else:
            camera = [filename.replace('.png', '.txt').replace('.jpg', '.txt') for filename in image_list]
            pass
        dataset = InferenceDataset(options, config, image_list=image_list, camera=camera)
        pass

    print('the number of images', len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    data_iterator = tqdm(dataloader, total=len(dataset))

    specified_suffix = options.suffix
    with torch.no_grad():
        detectors = []
        for method in options.methods:
            if method == 'f':
                options.suffix = specified_suffix if specified_suffix != '' else ''
                detectors.append(('final', PlaneRCNNDetector(options, config, modelType='final')))
                pass
            continue
        pass

    for name, detector in detectors:
        for sampleIndex, sample in enumerate(data_iterator):
            if options.testingIndex >= 0 and sampleIndex != options.testingIndex:
                if sampleIndex > options.testingIndex:
                    break
                continue
            input_pair = []
            camera = sample[30][0].cuda()
            for indexOffset in [0, ]:
                images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()

                masks = (gt_segmentation == torch.arange(gt_segmentation.max() + 1).cuda().view(-1, 1, 1)).float()
                input_pair.append({'image': images, 'depth': gt_depth, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera, 'plane': planes[0], 'masks': masks, 'mask': gt_masks})
                continue

            if sampleIndex >= options.numTestingImages:
                break

            with torch.no_grad():
                p_fea, e_fea, f_fea = detector.detect(sample)
                pass
            

            fea_dir = os.path.join(options.test_dir, str(sampleIndex)+'feature')
            if not os.path.exists(fea_dir):
                os.system("mkdir -p "+fea_dir)
            # image
            images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
            images = unmold_image(images, config)
            image = images[0]
            cv2.imwrite(fea_dir + '/' + 'Aimage' + '_' + str(0) + '.png', image[80:560])
            
            # edge feature
            print('obtained from edge extraction')
            print(e_fea[0].shape, e_fea[1].shape, e_fea[2].shape)
            print('')
            draw_feature(e_fea[0][:,:,20:140], fea_dir+'/L2', 'edge')
            draw_feature(e_fea[1][:,:,10:70], fea_dir+'/L3', 'edge')
            draw_feature(e_fea[2][:,:,5:35], fea_dir+'/L4', 'edge')
            
            print('obtained from multiscale')
            print(f_fea[0].shape, f_fea[1].shape, f_fea[2].shape)
            print('')
            draw_feature(f_fea[0][:,:,20:140], fea_dir+'/L2', 'multiscale')
            draw_feature(f_fea[1][:,:,10:70], fea_dir+'/L3', 'multiscale')
            draw_feature(f_fea[2][:,:,5:35], fea_dir+'/L4', 'multiscale')
            
            print('after resolution adaptation')
            print(p_fea[0].shape, p_fea[1].shape, p_fea[2].shape)
            print('')
            draw_feature(p_fea[0][:,:,20:140], fea_dir+'/L2', 'adaptation')
            draw_feature(p_fea[1][:,:,10:70], fea_dir+'/L3', 'adaptation')
            draw_feature(p_fea[2][:,:,5:35], fea_dir+'/L4', 'adaptation')
            
            if sampleIndex >= options.numTestingImages:
                break
            continue



if __name__ == '__main__':
    args = parse_args()

    if args.dataset == '':
        args.keyname = 'evaluate_visfeature'
    else:
        args.keyname = args.dataset
        pass
    args.test_dir = 'test/' + args.keyname
    
    # args.testingIndex=0
    
    if args.testingIndex >= 0:
        args.debug = True
        pass
    if args.debug:
        args.test_dir += '_debug'
        args.printInfo = True
        pass


    if not os.path.exists(args.test_dir):
        os.system("mkdir -p %s"%args.test_dir)
        pass

    # if args.debug and args.dataset == '':
    #     os.system('rm ' + args.test_dir + '/*')
    #     pass

    evaluate(args)
