import cv2
import os
import numpy as np
import time

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
def cut_pic(file_name):
    tcrop=False
    img = cv2.imread(file_name)
    h, w, _ = img.shape
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化处理
    ret, binary = cv2.threshold(GrayImage, 15, 255, cv2.THRESH_BINARY)  # 图片二值化,灰度值大于40赋值255，反之0
    threshold = 10  # 噪点阈值
    contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 直接填充二值图
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算轮廓所占面积
        if area < threshold:  # 将area小于阈值区域填充背景色
            cv2.drawContours(binary, [contours[i]], -1, 0, thickness=-1)  # 原始图片背景0
            continue

    # print(binary.shape, end=' ')
    # 得到所有目标点 横坐标、纵坐标
    edges_x, edges_y = np.where(binary == 255)
    if len(edges_x)==0:
        return trangle_crop(file_name)
    else:
        top = min(edges_x)  # 上边界
        bottom = max(edges_x)  # 下边界
        height = bottom - top  # 高度

        left = min(edges_y)  # 左边界
        right = max(edges_y)  # 右边界
        width = right - left  # 宽度
        return img[top:top + height, left:left + width]
    # print((height, width))
    # 返回剪切后的图




def trangle_crop(img_path):
    # print(img_path)
    raw = Image.open(img_path)
    W, H = raw.size  # Get dimensions
    if H > W:
        newH = W
        newW = W
    else:
        newH = H
        newW = H

    left = (W - newW) // 2
    top = (H - newH) // 2
    right = (W + newW) // 2
    bottom = (H + newH) // 2

    im = raw.crop((left, top, right, bottom))

    im=im.resize((800, 800))
    # img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return im

def trangle_resize(img_path):
    raw = Image.open(img_path)
    im=raw.resize((800, 800))
    # img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return im
#
# basedir= '/home/user6/Dataset/RFMiD_extra/val'
# save_file='/home/user6/Dataset/RF800/val'
# files=os.listdir(basedir)
# for file in files:
#     img_file=os.path.join(basedir,file)
#     save_lab_file = os.path.join(save_file, file)
#     if not os.path.exists(save_lab_file):
#         os.makedirs(save_lab_file)
#     imgs = os.listdir(img_file)
#     for img in tqdm(imgs):
#         img_single_file = os.path.join(img_file, img)
#         save_single_file = os.path.join(save_lab_file, img)
#         # save_single_file = os.path.join(save_lab_file, img).replace("png","png")
#         # print(save_single_file)
#         if(os.path.exists(save_single_file)):
#             continue
#
#         img_copy=trangle_crop(img_single_file)
#         # img_copy = trangle_resize(img_single_file)
#
#         # cv2.imwrite(save_single_file, img_copy)
#         img_copy.save(save_single_file)


##################文件夹未按标注区分时###########################

basedir= '/home/user6/Dataset/messidor_data'
save_file='/home/user6/Dataset/Messidor_Crop'
if not os.path.exists(save_file):
    os.makedirs(save_file)
imgs = os.listdir(basedir)
for img in tqdm(imgs):
    img_single_file = os.path.join(basedir, img)
    save_single_file = os.path.join(save_file, img).replace("tif","png")
    # save_single_file = os.path.join(save_single_file , img).replace("png","png")
    # print(save_single_file)

    img_copy=trangle_crop(img_single_file)
    # img_copy = trangle_resize(img_single_file)

    # cv2.imwrite(save_single_file, img_copy)
    img_copy.save(save_single_file)

# basedir="/home/data/PAC/PAC_data/test/38901_left.jpeg"
# save_file="../Dataset/DR_SumData/PAC_prepo/test_all/0/38901_left.jpeg"
# img_copy=cut_pic(basedir)
# cv2.imwrite(save_file, img_copy)

