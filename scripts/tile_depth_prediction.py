import os
import cv2
import numpy as np
from tqdm import tqdm

IMGROOT = '/home/ubuntu6/wzc/PlaneSeg/PlaneRCNN/test/eval_res_concat_BACK_nyu_2'
PRROOT = '/home/ubuntu6/wzc/PlaneSeg/PlaneRCNN/test/eval_2nd_5epoch_nyu'
PRPSROOT = '/home/ubuntu6/wzc/PlaneSeg/PlaneRCNN/test/eval_res_concat_BACK_nyu'
SAVE = '/home/ubuntu6/wzc/PlaneSeg/PlaneRCNN/vis_results/1_depth_prediction'
for i in tqdm(range(160, 654)):
    try:
        rawimg = cv2.imread(os.path.join(IMGROOT, str(i)+'_image_0.png'))
        gtimg = cv2.imread(os.path.join(IMGROOT, str(i)+'_plane_Depth_gt.png'))
        PRimg = cv2.imread(os.path.join(PRROOT, str(i)+'_depth_0_final.png'))
        PRPSimg = cv2.imread(os.path.join(PRPSROOT, str(i)+'_depth_0_final.png'))
        
        row = np.zeros((480, 640*4+10*3,3))
        row[:,0:640] = rawimg
        row[:,640+10:2*640+10] = PRimg
        row[:,2*640+2*10:3*640+2*10] = PRPSimg
        row[:,3*640+3*10:4*640+3*10] = gtimg
        
        cv2.imwrite(os.path.join(SAVE, str(i)+'_compare.png'), row)
    except:
        pass