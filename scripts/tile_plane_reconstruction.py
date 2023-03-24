import os
import cv2
import numpy as np
from tqdm import tqdm
import os, shutil
PRROOT = '/home/ubuntu6/wzc/PlaneSeg/PlaneRCNN/test/PlaneRCNNwoPlaneSeg_visualization300'
PRPSROOT = '/home/ubuntu6/wzc/PlaneSeg/PlaneRCNN/test/PlaneRCNNwPlaneSeg_visualization300'
SAVE = '/home/ubuntu6/wzc/PlaneSeg/PlaneRCNN/vis_results/2_plane_reconstruction'
if os.path.exists(SAVE):
    shutil.rmtree(SAVE)
os.makedirs(SAVE)

PAD = 40

def resize_figure(img):
    ini_size = img.shape
    x = 0
    xx = img.shape[0]
    y = 0
    yy = img.shape[1]
    for channel in range(img.shape[2]):
        for i in np.arange(0, img.shape[0], 1):
            if img[i, :, channel].sum() != 255*img.shape[1]:
                x = max(x, i)
                break
        for i in np.arange(img.shape[0] - 1, -1, -1):
            if img[i, :, channel].sum() != 255*img.shape[1]:
                xx = min(xx, i)
                break
        for j in np.arange(0, img.shape[1], 1):
            if img[:, j, channel].sum() != 255*img.shape[0]:
                y = max(y, j)
                break
        for j in np.arange(img.shape[1] - 1, -1, -1):
            if img[:, j, channel].sum() != 255*img.shape[0]:
                yy = min(yy, j)
                break
    cutted_res = img[max(0,x - PAD):min(img.shape[0],xx + PAD), max(0, y - PAD):min(img.shape[1], yy + PAD), :]
    return cutted_res

for i in tqdm(range(300)):
    # try:
    rawimg = cv2.imread(os.path.join(PRROOT, str(i)+'_image_0.png'))
    PRdep = cv2.imread(os.path.join(PRROOT, str(i)+'_depth_0_final.png'))
    PRseg = cv2.imread(os.path.join(PRROOT, str(i)+'_segmentation_0_final.png'))
    PRrec = cv2.imread(os.path.join(PRROOT, str(i)+'_model_0_final.png'))
    assert PRrec.shape==(950, 1720, 3)
    PRrec = cv2.resize(resize_figure(PRrec[100:850,360:1360]), (640,480))
    
    
    PRPSdep = cv2.imread(os.path.join(PRPSROOT, str(i)+'_depth_0_final.png'))
    PRPSseg = cv2.imread(os.path.join(PRPSROOT, str(i)+'_segmentation_0_final.png'))
    PRPSrec = cv2.imread(os.path.join(PRPSROOT, str(i)+'_model_0_final.png'))
    assert PRPSrec.shape==(950, 1720, 3)
    PRPSrec = cv2.resize(resize_figure(PRPSrec[100:850,360:1360]), (640,480))

    row = np.zeros((480, 640*7+10*6,3))
    row[:,0:640] = rawimg
    row[:,640+10:2*640+10] = PRdep
    row[:,2*640+2*10:3*640+2*10] = PRseg
    row[:,3*640+3*10:4*640+3*10] = PRrec
    row[:,4*640+4*10:5*640+4*10] = PRPSdep
    row[:,5*640+5*10:6*640+5*10] = PRPSseg
    row[:,6*640+6*10:7*640+6*10] = PRPSrec
    
    cv2.imwrite(os.path.join(SAVE, str(i)+'_compare.png'), row)
    # except:
    #     print(i)