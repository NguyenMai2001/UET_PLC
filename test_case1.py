# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:02:43 2021

@author: Administrator
"""

import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

#1
def find_location_crop(event, x, y, flags, param):
    f = open(resource_path('data/config/location_crop.txt'), 'a')
    if event == cv2.EVENT_LBUTTONDOWN:
        f.write(str(x) + "\n")
        f.write(str(y) + "\n")
    f.close()

#2
def crop_image(image):
    a = []
    f = open(resource_path('data/config/location_crop.txt'), 'r+')
    x1 = int(f.readline())
    y1 = int(f.readline())
    x2 = int(f.readline())
    y2 = int(f.readline())
    crop = image[y1:y2, x1:x2, :]
    # cv2.imwrite(resource_path('data/img_check_crop/{}.jpg'.format(i)), crop)
    a.append(crop)
    f.close()
    return a

#3
# def calc_mean(image):
#     return np.mean(image, axis=(0, 1))

# def calc_mean_all(image_list):
#     a = []
#     for i in range(4):
#         # img = cv2.imread(resource_path('data/img_check_crop/{}.jpg'.format(i)))
#         img = image_list[i]
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         a.append(calc_mean(img))
#     return a

def check(mean):
    # img=cv2.imread("file_camimg/d1.jpg")
    # img_w=cv2.imread("file_camimg/l1") #demo
    image = cv2.resize(mean,(1920,1080))
    start_row,start_col= 410,783
    end_row,end_col= 635,1170

    cropped=image[start_row:end_row,start_col:end_col]
    gray_tray_1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    histr1 = cv2.calcHist([gray_tray_1], [0], None, [256], [0, 256])

    # print(histr2)
    # print(histr2)

    # plt.subplot(121)
    # plt.imshow(gray_tray_1)
    # plt.subplot(122)
    # plt.plot(histr1)
    # plt.show()
    if max(histr1) > 20000:
        return 0 #co lech
    return 1

#demo chek lech
img=cv2.imread("file_camimg/d2.jpg")
img_w=cv2.imread("file_camimg/l4.jpg")

dem = check(img)
if dem:
    print("vi tri dung")
else:
    print("lech")










# cap = cv2.VideoCapture(1)
# cap.set(3, 1280)
# cap.set(4, 720)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     if key == ord('s'):
#         cv2.imwrite('preview.jpg', frame)

# cap.release()
# cv2.destroyAllWindows()

# image = cv2.imread("preview.jpg")

# cv2.namedWindow("image")
# cv2.setMouseCallback("image", find_location_crop)
# while True:
#     cv2.imshow("image", image)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cv2.destroyAllWindows()

# crop_list = crop_image(image)

# mean = calc_mean_all(crop_list)
# print(mean)
# print(check(mean))
