#!/usr/bin/env python3

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QSizePolicy

import cv2
import numpy as np
import sys
import time
import os
import math


from connectPLC import PLC




def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Thread(QThread):
    progress = pyqtSignal()

    def run(self):
        while True:
            self.progress.emit()
            time.sleep(0.1)

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        

        # Declare Main Variable
        self.total = 0
        self.number_tested = 0
        self.number_success = 0
        self.number_error1 = 0
        self.number_error2 = 0
        self.number_error3 = 0
        self.count = 0

        self.cap_detect = any
        self.cap_check = any
        self.get_cap_detect = False
        self.get_cap_check = False

        self.Controller = PLC()
        self.command = ""
        self.error_val = ""

    def update_data(self, data):
        

        # Send Data to PLC -> Send Command to PLC -> Grip
        self.Controller.data = data
        self.Controller.sendData()
        self.Controller.command = "Grip"
        self.Controller.sendCommand()

    # Hàm cập nhật giờ   
    
    def main_process(self):
        if self.command == "Idle":
            # Kiểm tra xem đã nhận Camera Check chưa
            if self.get_cap_detect == True:

                # Reset Main Variables
                self.total = 0
                self.number_tested = 0
                self.number_success = 0
                self.number_error1 = 0
                self.number_error2 = 0
                self.number_error3 = 0
                self.count = 0

        elif self.command == "Detect":
            # Kiểm tra xem đã nhận Camera Check chưa
            if  True:

                # Detect YES/NO
                #Tra ve mang 192 gia tri 0 - 1
                result = np.zeros(192, dtype=int)
                print(result)

                check_yes = np.array([0, 20, 90, 101, 180])
                print(check_yes)

                for i in range(192):
                    for j in range(check_yes.size):
                        if i == check_yes[j]: 
                            result[i] = 1

                
                # Gửi kết quả Detect YES/NO cho PLC và Table  
                self.update_data(result)
                
                self.command = "Done_detect"
            
        elif self.command == "Check":
            # Kiểm tra xem đã nhận Camera Check chưa
            if self.get_cap_check == True:
                
                # Kiểm tra Jig
                checkOnJig = 1

                # Nếu không có linh kiện trên Jig
                if checkOnJig == 0:
                    self.command = "SOS"
                
                # Nếu có linh kiện trên Jig
                else:
                    
                    checkAlign = 1
                    
                    # Kết quả trả về linh kiện không lệch
                    if checkAlign:
                        # Đổi State -> Gửi State mới cho PLC
                        self.Controller.command = "ok_for_jig"
                        self.Controller.sendCommand()
                    
                    # Kết quả trả về linh kiện lệch
                    else:
                        # Đổi State -> Gửi State mới cho PLC
                        self.Controller.command = "SOS"
                        self.Controller.sendCommand()
                    
                    # Đổi State: nhận dự liệu từ bộ test
                    self.command = checkError.err_Check()

        # Nhận kết quả từ PLC -> Cập nhật bảng số liệu -> Gửi lệnh cho PLC tiếp tục gắp linh kiện mới -> Chờ tay gắp


        # elif self.command == "1": #Khong loi
        #     self.update_statistic(self.command)
        #     self.Controller.command = "Grip"
        #     self.Controller.sendCommand()
        #     self.command = "Wait"
        # elif self.command == "0": #Loi 
        #     self.update_statistic(self.command)
        #     self.Controller.command = "Grip" 
        #     self.Controller.sendCommand()
        #     self.command = "Wait"
        # elif self.command == "-1":
        #     self.update_statistic(self.command)
        #     self.Controller.command = "Grip"
        #     self.Controller.sendCommand()
        #     self.command = "Wait"
        # elif self.command == "404":
        #     self.update_statistic(self.command)
        #     self.Controller.command = "Grip"
        #     self.Controller.sendCommand()
        #     self.command = "Wait"
        
        # Kết thúc -> Xuất ra thông báo
        elif self.command == "Report":
            print("Report")

    # Loop Get Command from PLC
    def get_command(self):
        self.command = self.Controller.queryCommand()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
