import open3d as o3d
import os
import time
import numpy as np
from natsort import natsorted
# import pyautogui
# import cv2
from tqdm import tqdm
from PIL import Image,ImageGrab

# visualization of point clouds.
path="/home/ubuntu6/wzc/PlaneSeg/PlaneRCNN/test/PlaneRCNNwPlaneSeg_visualization300"
dirs=os.listdir(path)


def visPcd(pcd,picname): # 需要open3d,time库,默认暂停2秒，暂停时间在函数内设置
    # 创建可视化窗口并显示pcd
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(2)
    # 设置窗口存在时间,根据需要自行更改
    # 截取屏幕

    # img = pyautogui.screenshot(region=[0,0,100,100]) # x,y,w,h
    # # img.save('screenshot.png')
    # img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    x=100
    y=50
    width=1820
    height=1000
    img = ImageGrab.grab(bbox=(x, y, width, height))
    Image.fromarray(np.uint8(img))
    img.save(picname)

    # wait time
    time.sleep(2)
    # 关闭窗口
    vis.destroy_window()

for i in tqdm(natsorted([x for x in dirs if x[-3:]=="ply"])):
    file_type=i[-3:]
    assert file_type=="ply"
    ply_path=os.path.join(path, i)
    print("path:",ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    # o3d.visualization.draw_geometries([pcd])
    pic_name=os.path.join(path, i[:-4]+".png")
    print("pic_name:",pic_name)
    visPcd(pcd,pic_name)
