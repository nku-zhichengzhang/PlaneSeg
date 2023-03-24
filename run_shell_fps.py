import os
from time import sleep

PID = 20610
x = 0

# sleep(3600*1.3)
while x == 0:
    x = os.system('nvidia-smi | grep %d' % PID)
    # sleep(30)

# os.system('which python')
# os.system('python train_planercnn_ablation_edge.py --restore=2 --suffix=warping_refine')
os.system('python eval_fps_orig.py --methods=f --suffix=warping_refine')
os.system('python eval_fps_ours.py --methods=f --suffix=warping_refine')
os.system('python eval_fps_page.py --methods=f --suffix=warping_refine')
os.system('python eval_fps_scrn.py --methods=f --suffix=warping_refine')
os.system('python eval_fps_ce2p.py --methods=f --suffix=warping_refine')
os.system('python eval_fps_nldf.py --methods=f --suffix=warping_refine')