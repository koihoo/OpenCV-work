import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

img_dir = r'D:/210825/0910/bubingpian'
new_path = r'D:\210825\cut'


def get_feature(img):
    h = 224
    w = int(img.shape[1] / img.shape[0] * h)
    img = cv2.resize(img, (w, h))
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_img = hsv_img[:, :, 2]
    feature = np.mean(v_img,axis=1)
    line = np.argmax(feature)
    return line
for img in os.listdir(img_dir):
    filename = img
    img_path = os.path.join(img_dir,img)
    img = cv2.imread(img_path)
    feature = get_feature(img)
    img2 = Image.open(img_path)
    (H,W) = img2.size
    # img.show()
    left = 0
    top = feature -56 if feature >=56 else 0
    right = W
    bottom =  feature + 56 if feature <=H else H
    box1 = (left, top, right, bottom)  # 设置左、上、右、下的像素
    image1 = img2.crop(box1)  # 图像裁剪
    image1.save(r"D:\210825\cut\bubingpian-3" + '\\' + filename)  # 存储裁剪得到的图像


