import cv2
import numpy as np
import sys
import time
import os
import math
from matplotlib import pyplot as plt

import checkAlign
from connectPLC import PLC
from detectYesNo import check_chess
from detectYesNo import Detect
from checkOnJig import CheckOn


# def setup_camera(self):
#         self.cap_detect = cv2.VideoCapture(2)
#         # cv2.imshow(self.cap_detect)
#         # cv2.waitKey(0)
#         # Khai báo USB Camera Detect Config
#         self.get_cap_detect = True
        
#         self.cap_check = cv2.VideoCapture(0) # Khai báo USB Camera Check Config
#         self.cap_check.set(3, 1920)
#         self.cap_check.set(4, 1080)
#         # cv2.imshow(self.cap_check)
#         # cv2.waitKey(0)
#         self.get_cap_check = True
#         self.cap_detect.set(3, 1920)
#         self.cap_detect.set(4, 1080)
    
    
def main_process():
    cap_detect = cv2.VideoCapture(0)
    # cv2.imshow(self.cap_detect)
    # cv2.waitKey(0)
    # Khai báo USB Camera Detect Config
    get_cap_detect = True
    
    cap_check = cv2.VideoCapture(1) # Khai báo USB Camera Check Config
    cap_check.set(3, 1920)
    cap_check.set(4, 1080)
    # cv2.imshow(self.cap_check)
    # cv2.waitKey(0)
    get_cap_check = True
    cap_detect.set(3, 1920)
    cap_detect.set(4, 1080)
    # self.cap_detect.set(3, 1920)
    # self.cap_detect.set(4, 1080)
    
    ret, image = cap_detect.read()
    print("ret", ret)
    # image = cv2.resize(image, (int(717 * self.width_rate), int(450 * self.height_rate)), interpolation = cv2.INTER_AREA) # Resize cho Giao diện
    plt.imshow(image)
    plt.show()
    
    ret, image1 = cap_check.read() # Lấy dữ liệu từ camera
    # plt.subplot(2,1)
    # plt.imshow(image)
    plt.imshow(image1, cmap='gray')
    plt.show()

    #CHECK TRAY
    detect = Detect()
    img = check_chess(image)
    detect.rotated(img)
    detect.image = cv2.imread('rotated_image.jpg', cv2.IMREAD_GRAYSCALE)
    # plt.imshow(detect.image, cmap="gray")
    # plt.show()
    detect.thresh()

    # Detect YES/NO
    result = detect.check(detect.crop_tray_1)
    result = np.append(result, detect.check(detect.crop_tray_2))
    result = np.append(result, detect.check(detect.crop_tray_3))
    result = np.append(result, detect.check(detect.crop_tray_4))
    print(result)
    # self.update_detect_image(resize_img)

    # self.update_check_image(image1)

    #CHECK LECH

    # checkOn = CheckOn()
    cv2.imwrite('checkjig.jpg', image1)
    img_check = cv2.imread('checkjig.jpg', cv2.IMREAD_GRAYSCALE)
    check = checkAlign.check(img_check)
    plt.imshow(img_check, cmap='gray')
    plt.show()
    print(check)

main_process()