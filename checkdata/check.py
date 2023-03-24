import numpy as np
import cv2

from torch.utils.data import DataLoader

from config import InferenceConfig
import os
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
from tqdm import tqdm
from options import parse_args

'''1
preds_edge = np.load('edge.npy')[0][1]
target = np.load('target.npy')
print(preds_edge)
# cv2.imshow('edge', preds_edge)
# cv2.waitKey()

all_statistics = np.load('../logs/all_statistics.npy')

indice_path = '../../new_selected_sceneImageIndices_clear.npy'
clear_indices = np.load(indice_path)

all_statistics = np.load('../logs/all_statistics.npy')
'''
def evaluate(options):
    config = InferenceConfig(options)
    config.FITTING_TYPE = options.numAnchorPlanes
    dataset = PlaneDataset(options, config, split='test', random=False, load_semantics=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for sampleIndex,_ in enumerate(tqdm(dataloader)):
        if sampleIndex >= options.numTestingImages:
            break
        continue
    print()

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


    if not os.path.exists(args.test_dir):
        os.system("mkdir -p %s"%args.test_dir)
        pass

    if args.debug and args.dataset == '':
        os.system('rm ' + args.test_dir + '/*')
        pass

    evaluate(args)