import cv2
import numpy as np
import sys
import time
import os
import math
from matplotlib import pyplot as plt
import codecs
import serial
import time
import wheel

import checkAlign
from connectPLC import PLC
from detectYesNo import check_chess
from detectYesNo import Detect
from checkOnJig import CheckOn
# import Send_data_func as f

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)


class demov1(object):
    def __init__(self):
        super().__init__()

        # Declare Main Variable
        
        self.cap_detect = any
        self.cap_check = any
        self.get_cap_detect = False
        self.get_cap_check = False

        #NUC - PLC
        self.Controller = PLC()
        self.command = ''
        self.status_cam_checked = ''
        self.status_cam_inJig = ''
        self.jig_signal = False
        self.prev_command = ''

        self.checkError = " "
    
    def setup_camera(self):
        self.cap_detect = cv2.VideoCapture(1) # Khai báo USB Camera Detect Config
        self.get_cap_detect = True
        self.cap_detect.set(3, 1920)
        self.cap_detect.set(4, 1080)
         # cv2.imshow(self.cap_detect)
        # cv2.waitKey(0)

        self.cap_check = cv2.VideoCapture(0) # Khai báo USB Camera Check Config
        self.get_cap_check = True
        
        self.cap_check.set(3, 1920)
        self.cap_check.set(4, 1080)
        # cv2.imshow(self.cap_check)
        # cv2.waitKey(0)

    # Loop Get Command from PLC
    def get_command(self):
        self.command = self.Controller.queryCommand()
        print(self.command)
        if self.command == "Done_detect":
            self.prev_command = 'Done_detect'
        self.status_cam_checked = self.Controller.status_cam_checked()
        self.status_cam_inJig = self.Controller.status_cam_in_jig()
        self.jig_signal = self.Controller.jig_Signal()

    # def get_status_cam_check(self):
    #     self.status_cam_checked = self.Controller.status_cam_checked()
    
    # def get_status_cam_inJig(self):
    #     self.status_cam_inJig = self.Controller.status_cam_in_jig()

    # def get_jig_signal(self):
    #     self.jig_signal = self.Controller.jig_Signal()

    # def check_error(self):
    #     self.check_error = err_Check("1")
    def main_process(self):
        if self.command == 'Detect':
            if self.get_cap_detect == True:
                ret, image = self.cap_detect.read()
                # image = cv2.resize(image, (int(717 * self.width_rate), int(450 * self.height_rate)), interpolation = cv2.INTER_AREA) # Resize cho Giao diện
                #plt.imshow(image)
                #plt.show()

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
                # plt.imshow(detect.image, cmap="gray")
                # plt.show()
                #up date result to PLC
                self.Controller.data = result
                self.Controller.sendData()
            
                self.command = 'Done_detect'
                self.Controller.sendCommand(self.command)
            
        elif self.command == 'Check' and self.prev_command == 'Done_detect':
            # print("Check")
            if self.get_cap_check == True:
                ret, image1 = self.cap_check.read() # Lấy dữ liệu từ camera
                # plt.subplot(2,1)
                # plt.imshow(image)
                #plt.imshow(image1, cmap='gray')
                #plt.show()

                cv2.imwrite('checkjig.jpg', image1)
                img_check = cv2.imread('checkjig.jpg', cv2.IMREAD_GRAYSCALE)
                check = checkAlign.check(img_check)
                print(check)
                # plt.imshow(img_check, cmap='gray')
                # plt.show()

                if check:
                    self.status_cam_inJig = 'Ok_for_jig'
                    self.Controller.send_status_cam_inJig(self.status_cam_inJig)
                    
                    self.jig_signal = self.Controller.status_cam_in_jig()
                    #Test on Jig
                    if self.jig_signal: #da giap Jig
                        #self.checkError = f.err_Check("a")
                        #====================================================================#
                        time.sleep(1)
                        user = "1"
                        arduino.write(bytes(user,'utf-8'))
                        time.sleep(5)
                        data = arduino.readline()
                        print(data)
                        #Change byte to string
                        errCheck_str = data.decode(encoding="utf-8")
                        #print(len(errCheck_str))
                        errCheck = []
                        if(len(errCheck_str)==3):
                            index = 0
                        else: 
                            if(len(errCheck_str)==9):
                                errCheck.append("0")
                                for i in range(len(errCheck_str)):
                                    errCheck.append(errCheck_str[i])
                            else:
                                for i in range(len(errCheck_str)):
                                    errCheck.append(errCheck_str[i])

                            # print("Err_Check: ",errCheck)
                            if errCheck[1] == "1": index = 2
                            if errCheck[7] == "1": index = 8
                            if errCheck[2] == "1": index = 4
                            if errCheck[4] == "1": index = 5
                            if errCheck[6] == "1": index = 6
                            if errCheck[5] == "1": index = 7
                            if errCheck[3] == "1": index = 3
                            if errCheck[0] == "1": index = 1
                        dispMap =["OK" , " I2C" , "DF_S" ,"AF_D" ,"FIXD" ,"EMPT" ,"DATA" ,"AWB" ," CRC"]
                        Error = dispMap[index]
                        print("loi hien tai:", Error)
                #================================================================================#
                        self.checkError = Error

                        if self.checkError == "OK":
                            self.status_cam_checked = 'OK'
                            print(Error)
                            self.Controller.send_status_cam_check(self.status_cam_checked)
                        else:
                            self.status_cam_checked = 'NG'
                            print(Error)
                            self.Controller.send_status_cam_check(self.status_cam_checked)
                    
                else:
                    self.status_cam_inJig = 'Skeef'
                    self.Controller.send_status_cam_inJig(self.status_cam_inJig)
            self.prev_command = 'check'
                


if __name__ == '__main__':
    ex = demov1()
    ex.setup_camera()
    while True:
        t1 = time.time()
        ex.get_command()
        ex.main_process()
        ex.command = ''
        print(time.time() - t1)
                




    


        
    

