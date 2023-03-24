import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from torch import optim
from torch.utils.data import DataLoader
# from torchstat import stat
from torchvision.models import resnet50
from tqdm import tqdm
import numpy as np
import cv2
import sys

from models.model import *
from models.refinement_net import *
from models.modules import *
from datasets.plane_stereo_dataset import *

from utils import *
from visualize_utils import *
from evaluate_utils import *
from options import parse_args
from config import PlaneConfig
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def eval_our(options):


    config = PlaneConfig(options)
    std = resnet50()
    model = MaskRCNN_edge_fpn_resolution_paper(config)
    resnet = ResNet("resnet101", stage5=True, numInputChannels=config.NUM_INPUT_CHANNELS)
    C1, C2, C3, C4, C5 = resnet.stages()

    edge_module = Edge_Module()
    fpn_module = FPN_edge(C1, C2, C3, C4, C5, out_channels=256, bilinear_upsampling=config.BILINEAR_UPSAMPLING)
    channel = 256
    pn_transform2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 3, 1, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 1, 1, 0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        # Bottleneck(256, 256 // 4),
                                        ) ##
    pn_transform3 = nn.Sequential(nn.Conv2d(256, 256, 1, 1, 0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 1, 1, 0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        # Bottleneck(256, 256 // 4),
                                        ) ##
    pn_transform4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1, 0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        # Bottleneck(256, 256 // 4),
                                        ) ##
    Res = nn.Sequential(pn_transform2,pn_transform3,pn_transform4)
    # print('std')
    # count_parameters(std)
    print('backbone')
    count_parameters(resnet)
    print('edge')
    count_parameters(model.edge_layer)
    print('fpn')
    count_parameters(fpn_module)
    print('resAda')
    count_parameters(Res)
    count_parameters(model)

    return


if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'planercnn'

    args.keyname += '_' + args.anchorType
    if args.dataset != '':
        args.keyname += '_' + args.dataset
        pass
    if args.trainingMode != 'all':
        args.keyname += '_' + args.trainingMode
        pass
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname + '_ablation_edge_fpn_resolution'
    args.test_dir = 'test/' + args.keyname

    if False:
        writeHTML(args.test_dir, ['image_0', 'segmentation_0', 'depth_0', 'depth_0_detection', 'depth_0_detection_ori'], labels=['input', 'segmentation', 'gt', 'before', 'after'], numImages=20, image_width=160, convertToImage=True)
        exit(1)
        
    os.system('rm ' + args.test_dir + '/*.png')
    print('keyname=%s task=%s started'%(args.keyname, args.task))

    eval_our(args)