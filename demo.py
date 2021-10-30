#!/usr/bin/env python3

# from CODE.detectYesNo import check_chess
from PIL.Image import Image
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

# import checkAlign
#from connectPLC import PLC
# from detectYesNo import check_chess
# from detectYesNo import Detect
# from checkOnJig import CheckOn


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

class Query(QThread):
    progress = pyqtSignal()

    def run(self):
        while True:
            self.progress.emit()
            time.sleep(0.5)

class Camera(QThread):
    setup = pyqtSignal()

    def run(self):
        self.setup.emit()

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # QT Config
        self.title = "PLC-UET"
        self.icon = QIcon(resource_path('data/icon/uet.png'))

        # Declare Main Variable
        self.total = 0
        self.number_tested = 0
        self.number_success = 0
        self.number_error1 = 0
        self.number_error2 = 0
        self.number_error3 = 0
        self.count = 0

        #Camera
        self.cap_detect = any
        self.cap_check = any
        self.get_cap_detect = False
        self.get_cap_check = False

        #NUC - PLC
        # self.Controller = PLC()
        # self.command = ''
        # self.status_cam_checked = ''
        # self.status_cam_inJig = ''
        # self.jig_signal = False

        # self.checkError = " "

        # Run QT
        self.initUI()
    
    def initUI(self):

        # Config Main Window
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.icon)
        self.setWindowState(Qt.WindowFullScreen)
        self.setStyleSheet("background-color: rgb(255, 255, 255);")

        # Config Auto Fit Screen Scale Variables
        self.sg = QDesktopWidget().screenGeometry()
        self.width_rate = self.sg.width() / 1920
        self.height_rate = self.sg.height() / 1080
        self.font_rate = math.sqrt(self.sg.width()*self.sg.width() + self.sg.height()*self.sg.height()) / math.sqrt(1920*1920 + 1080*1080)
        
        # Show MCNEX LOGO
        self.mcnex_logo = QLabel(self)
        self.mcnex_pixmap = QPixmap(resource_path('data/icon/mcnex.png')).scaled(181 * self.width_rate, 141 * self.width_rate, Qt.KeepAspectRatio)
        self.mcnex_logo.setPixmap(self.mcnex_pixmap)
        self.mcnex_logo.setGeometry(50 * self.width_rate, 1 * self.height_rate, 181 * self.width_rate, 141 * self.height_rate)
        
         # Show UET LOGO
        self.uet_logo = QLabel(self)
        self.uet_pixmap = QPixmap(resource_path('data/icon/uet.png')).scaled(111 * self.width_rate, 111 * self.width_rate, Qt.KeepAspectRatio)
        self.uet_logo.setPixmap(self.uet_pixmap)
        self.uet_logo.setGeometry(250 * self.width_rate, 10 * self.height_rate, 111 * self.width_rate, 111 * self.height_rate)

        # Show Title
        self.title_label = QLabel("HỆ THỐNG KIỂM TRA LINH KIỆN", self)
        self.title_label.setGeometry(400 * self.width_rate, 17 * self.height_rate, 1800 * self.width_rate, 95 * self.height_rate)
        font_title = QFont('', int(25 * self.font_rate), QFont.Bold)
        self.title_label.setFont(font_title)
        self.title_label.setStyleSheet("color: rgb(0, 0, 0);")

         # Show Current Time
        self.time_label = QLabel(self)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setGeometry(1470 * self.width_rate, 20 * self.height_rate, 430 * self.width_rate, 95 * self.height_rate)
        font_timer = QFont('', int(25 * self.font_rate), QFont.Bold)
        self.time_label.setFont(font_timer)
        timer = QTimer(self)
        timer.timeout.connect(self.updateTimer)
        timer.start(1000)
        self.time_label.setStyleSheet("color: rgb(0, 0, 0);")

        # Show Detect Camera
        # self.cam1_name = QLabel("DETECT CAMERA", self)
        # self.cam1_name.setGeometry(55 * self.width_rate, 127 * self.height_rate, 582 * self.width_rate, 60 * self.height_rate)
        # self.cam1_name.setAlignment(Qt.AlignCenter)
        # self.cam1_name.setStyleSheet("background-color: rgb(50, 130, 184);"
        #                             "color: rgb(255, 255, 255);"
        #                             "font: bold 14pt;")
        # self.cam1 = QLabel(self)
        # self.cam1.setGeometry(55 * self.width_rate, 185 * self.height_rate, 582 * self.width_rate, 410 * self.height_rate)
        # self.cam1.setStyleSheet("border-color: rgb(50, 130, 184);"
        #                         "border-width: 5px;"
        #                         "border-style: inset;")
        
        # Show Check Camera
        self.cam2_name = QLabel("CHECK CAMERA", self)
        self.cam2_name.setGeometry(1265 * self.width_rate, 127 * self.height_rate, 610 * self.width_rate, 60 * self.height_rate)
        self.cam2_name.setAlignment(Qt.AlignCenter)
        self.cam2_name.setStyleSheet("background-color: rgb(50, 130, 184);"
                                    "color: rgb(255, 255, 255);"
                                    "font: bold 14pt;")
        self.cam2 = QLabel(self)
        self.cam2.setGeometry(1265 * self.width_rate, 185 * self.height_rate, 610 * self.width_rate, 400 * self.height_rate)
        self.cam2.setStyleSheet("border-color: rgb(50, 130, 184);"
                                "background-color: rgb(220, 220, 220);"
                                "border-width: 5px;"
                                "border-style: inset;")

        # Set Font
        self.font = QFont('', int(14 * self.font_rate), QFont.Bold)

        
        # Trays Information
        self.tray = []
        for i in range(2):
            k = 2-i
            tray_name = QLabel("TRAY {}".format(k), self)
            tray_name.setGeometry((55 + 406*(i+1) - 5) * self.width_rate, 606 * self.height_rate, 359 * self.width_rate, 55 * self.height_rate)
            tray_name.setAlignment(Qt.AlignCenter)
            tray_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                    "color: rgb(255, 255, 255);"
                                    "font: bold 14pt;")
            table_margin = QLabel(self)
            table_margin.setGeometry((55 + 406*(i+1) - 5) * self.width_rate, 661 * self.height_rate, 359 * self.width_rate, 405 * self.height_rate)
            table_margin.setStyleSheet("border-color: rgb(50, 130, 184);"
                                        "border-width: 5px;"
                                        "border-style: inset;")
            table = QTableWidget(8, 6, self)
            table.setGeometry((55 + 406*(i+1)) * self.width_rate, 666 * self.height_rate, int(349 * self.width_rate) + 1, int(396 * self.height_rate) + 0.5)
            table.horizontalHeader().hide()
            table.verticalHeader().hide()
            for j in range(6):
                table.setColumnWidth(j, 58 * self.width_rate)
            for j in range(8):
                table.setRowHeight(j, 49 * self.height_rate)
            table.setFont(self.font)
            table.setStyleSheet("background-color:rgb(220, 220, 220);"
                                "color: rgb(255, 255, 255);")
            self.tray.append(table)

#=====
        tray_name = QLabel("NG {}".format(2), self)
        tray_name.setGeometry((55 + 406*0 - 5) * self.width_rate, 606 * self.height_rate, 359 * self.width_rate, 55 * self.height_rate)
        tray_name.setAlignment(Qt.AlignCenter)
        tray_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                "color: rgb(255, 255, 255);"
                                "font: bold 14pt;")
        table_margin = QLabel(self)
        table_margin.setGeometry((55 + 406*0 - 5) * self.width_rate, 661 * self.height_rate, 359 * self.width_rate, 405 * self.height_rate)
        table_margin.setStyleSheet("border-color: rgb(50, 130, 184);"
                                    "border-width: 5px;"
                                    "border-style: inset;")
        table = QTableWidget(8, 6, self)
        table.setGeometry((55 + 406*0) * self.width_rate, 666 * self.height_rate, int(349 * self.width_rate) + 1, int(396 * self.height_rate) + 0.5)
        table.horizontalHeader().hide()
        table.verticalHeader().hide()
        for j in range(6):
            table.setColumnWidth(j, 58 * self.width_rate)
        for j in range(8):
            table.setRowHeight(j, 49 * self.height_rate)
        table.setFont(self.font)
        table.setStyleSheet("background-color:rgb(220, 220, 220);"
                            "color: rgb(255, 255, 255);")
        self.tray.append(table)
        
        self.tray2 = []
        for i in range(2):
            tray_name = QLabel("TRAY {}".format(i+3), self)
            tray_name.setGeometry((55 + 406*(i+1)- 5) * self.width_rate, 127 * self.height_rate, 359 * self.width_rate, 55 * self.height_rate)
            tray_name.setAlignment(Qt.AlignCenter)
            tray_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                    "color: rgb(255, 255, 255);"
                                    "font: bold 14pt;")
            table_margin = QLabel(self)
            table_margin.setGeometry((55 + 406*(i+1) - 5) * self.width_rate, 181 * self.height_rate, 359 * self.width_rate, 405 * self.height_rate)
            table_margin.setStyleSheet("border-color: rgb(50, 130, 184);"
                                        "border-width: 5px;"
                                        "border-style: inset;")
            table = QTableWidget(8, 6, self)
            table.setGeometry((55 + 406*(i+1)) * self.width_rate, 186 * self.height_rate, int(349 * self.width_rate) + 1, int(396 * self.height_rate) + 0.5)
            table.horizontalHeader().hide()
            table.verticalHeader().hide()
            for j in range(6):
                table.setColumnWidth(j, 58 * self.width_rate)
            for j in range(8):
                table.setRowHeight(j, 49 * self.height_rate)
            table.setFont(self.font)
            table.setStyleSheet("background-color:rgb(220, 220, 220);"
                                "color: rgb(255, 255, 255);")
            self.tray2.append(table)

#==================================
        tray_name = QLabel("NG {}".format(1), self)
        tray_name.setGeometry((55 + 406*0 - 5) * self.width_rate, 127 * self.height_rate, 359 * self.width_rate, 55 * self.height_rate)
        tray_name.setAlignment(Qt.AlignCenter)
        tray_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                "color: rgb(255, 255, 255);"
                                "font: bold 14pt;")
        table_margin = QLabel(self)
        table_margin.setGeometry((55 + 406*0 - 5) * self.width_rate, 181 * self.height_rate, 359 * self.width_rate, 405 * self.height_rate)
        table_margin.setStyleSheet("border-color: rgb(50, 130, 184);"
                                    "border-width: 5px;"
                                    "border-style: inset;")
        table = QTableWidget(8, 6, self)
        table.setGeometry((55 + 406*0) * self.width_rate, 186 * self.height_rate, int(349 * self.width_rate) + 1, int(396 * self.height_rate) + 0.5)
        table.horizontalHeader().hide()
        table.verticalHeader().hide()
        for j in range(6):
            table.setColumnWidth(j, 58 * self.width_rate)
        for j in range(8):
            table.setRowHeight(j, 49 * self.height_rate)
        table.setFont(self.font)
        table.setStyleSheet("background-color:rgb(220, 220, 220);"
                            "color: rgb(255, 255, 255);")
        self.tray2.append(table)

        # Table Info Area        
        self.s_name = QLabel("INFORMATION", self)
        # self.s_name.setGeometry(1450 * self.width_rate, 127 * self.height_rate, 399 * self.width_rate, 60 * self.height_rate)
        self.s_name.setGeometry(1265 * self.width_rate, 605 * self.height_rate, 610 * self.width_rate, 60 * self.height_rate)

        self.s_name.setAlignment(Qt.AlignCenter)
        self.s_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                    "color: rgb(255, 255, 255);"
                                    "font: bold 14pt;")

        self.statistic_table = QTableWidget(5, 3, self)
        self.statistic_table.setGeometry(1265 * self.width_rate, 663 * self.height_rate, int(610 * self.width_rate) + 1, int(405 * self.height_rate) + 1)
        self.statistic_table.horizontalHeader().hide()
        self.statistic_table.verticalHeader().hide()
        self.statistic_table.setFont(self.font)
        self.statistic_table.setStyleSheet("color: rgb(0, 0, 0);"
                                            "background-color:rgb(220, 220, 220);"
                                            "text-align: center;"
                                            "border-width: 5px;"
                                            "border-style: inset;"
                                            "border-color: rgb(50, 130, 184);")
        for j in range(3):
            self.statistic_table.setColumnWidth(j, 200 * self.width_rate)
        for j in range(5):
            self.statistic_table.setRowHeight(j, 79 * self.height_rate)
        tested_item = QTableWidgetItem("TESTED")
        tested_item.setTextAlignment(Qt.AlignCenter)
        tested_item.setFont(self.font)
        self.statistic_table.setItem(0, 0, tested_item)

        success_item = QTableWidgetItem("SUCCESS")
        success_item.setTextAlignment(Qt.AlignCenter)
        success_item.setFont(self.font)
        self.statistic_table.setItem(1, 0, success_item)

        error1_item = QTableWidgetItem("NEED RETEST")
        error1_item.setTextAlignment(Qt.AlignCenter)
        error1_item.setFont(self.font)
        self.statistic_table.setItem(2, 0, error1_item)

        error2_item = QTableWidgetItem("CONNECTION ERROR")
        error2_item.setTextAlignment(Qt.AlignCenter)
        error2_item.setFont(self.font)
        self.statistic_table.setItem(3, 0, error2_item)

        error3_item = QTableWidgetItem("FAILURE")
        error3_item.setTextAlignment(Qt.AlignCenter)
        error3_item.setFont(self.font)
        self.statistic_table.setItem(4, 0, error3_item)

        # Exit Button
        self.exit_button = QPushButton(self)
        self.exit_pixmap = QPixmap(resource_path('data/icon/close.png')).scaled(100 * self.width_rate, 100 * self.width_rate, Qt.KeepAspectRatio)
        self.exit_icon = QIcon(self.exit_pixmap)
        self.exit_button.setIcon(self.exit_icon)
        self.exit_button.setIconSize(QSize(50, 50))
        self.exit_button.setGeometry(1878 * self.width_rate, -8 * self.height_rate, 50 * self.width_rate, 50 * self.height_rate)
        self.exit_button.setHidden(0)
        self.exit_button.setStyleSheet("border: none")
        self.exit_button.clicked.connect(self.close)

        # # Create Thread
        # self.camera_thread = Camera()
        # self.camera_thread.setup.connect(self.setup_camera)
        # self.main_thread = Thread()
        # self.main_thread.progress.connect(self.main_process)
        # self.plc_thread = Query()
        # self.main_thread.progress.connect(self.get_command)

        # # Run Thread
        # self.camera_thread.start()
        # self.main_thread.start()
        # self.plc_thread.start()


    #Hết giao diện ###########################################################################################################


    #Hàm stream CAMERA DETECT lên giao diện
    def update_detect_image(self, img):
        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.cam1.setPixmap(QPixmap.fromImage(convertToQtFormat))
    
    
    
    # Hàm stream CAMERA CHECK lên giao diện
    def update_check_image(self, img):
        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.cam2.setPixmap(QPixmap.fromImage(convertToQtFormat))
    
    
    # Hàm cập nhật giờ   
    def updateTimer(self):
        cr_time = QTime.currentTime()
        time = cr_time.toString('hh:mm AP')
        self.time_label.setText(time)

    #Khởi tạo bảng giá trị 
    def init_statistic(self):
        tested = QTableWidgetItem("{}".format(0) + " / {}".format(self.total))
        tested.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(0,1,tested)
        # ratio_tested = QTableWidgetItem("{} %".format(0))
        # ratio_tested.setTextAlignment(Qt.AlignCenter)
        # self.statistic_table.setItem(0,2,ratio_tested)

        success = QTableWidgetItem("{}".format(0) + " / {}".format(0))
        success.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(1,1,success)
        # ratio_success = QTableWidgetItem("{} %".format(0))
        # ratio_success.setTextAlignment(Qt.AlignCenter)
        # self.statistic_table.setItem(1,2,ratio_success)

        error1 = QTableWidgetItem("{}".format(0) + " / {}".format(0))
        error1.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(2,1,error1)
        # ratio_error1 = QTableWidgetItem("{} %".format(0))
        # ratio_error1.setTextAlignment(Qt.AlignCenter)
        # self.statistic_table.setItem(2,2,ratio_error1)

        error2 = QTableWidgetItem("{}".format(0) + " / {}".format(0))
        error2.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(3,1,error2)
        # ratio_error2 = QTableWidgetItem("{} %".format(0))
        # ratio_error2.setTextAlignment(Qt.AlignCenter)
        # self.statistic_table.setItem(3,2,ratio_error2)

        error3 = QTableWidgetItem("{}".format(0) + " / {}".format(0))
        error3.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(4,1,error3)
        # ratio_error3 = QTableWidgetItem("{} %".format(0))
        # ratio_error3.setTextAlignment(Qt.AlignCenter)
        # self.statistic_table.setItem(4,2,ratio_error3)
    
    #Hàm cập nhật số liệu về tổng số linh kiện có trên tray:
    def update_YesNo_data_to_table(self, data):
        
        # Update Data to Table --> Tính Total

        #Update Data to tray 1, tray 2
        c = 0
        for k in range(2):    
            for j in range(8):
                for i in range(6):
                    self.tray[1-k].setItem(j,i,QTableWidgetItem())
                    if(int(data[c])):
                        self.tray[1-k].item(j,i).setBackground(QColor(128, 255, 0)) #Nếu có linh kiện, đổi màu thành xanh
                        self.total += 1
                    else:
                        self.tray[1-k].item(j,i).setBackground(QColor(255,0, 0)) #Không có linh kiện, đổi màu thành đỏ
                        self.total += 1
                    c += 1
        
        #Up Data to Tray 3, tray 4
    
        for k in range(2):    
            for j in range(8):
                for i in range(6):
                    self.tray2[k].setItem(j,i,QTableWidgetItem())
                    if(int(data[c])):
                        self.tray2[k].item(j,i).setBackground(QColor(128, 255, 0)) #Nếu có linh kiện, đổi màu thành xanh
                        self.total += 1
                    else:
                        self.tray2[k].item(j,i).setBackground(QColor(255,0, 0)) #Không có linh kiện, đổi màu thành đỏ
                        self.total += 1
                    c += 1

    #Hàm cập nhật bảng số liệu _ cập nhật Information Table
    def update_information_table():
        self.number_tested += 1
    
    #xử lý ảnh
    def main_process(self):
        # Reset Main Variables
        self.total = 0
        self.number_tested = 0
        self.number_success = 0
        self.number_error1 = 0
        self.number_error2 = 0
        self.number_error3 = 0
        self.count = 0

        file_test = cv2.imread('Camera_test/tray/tray (14).jpg')
        self.update_check_image(file_test)

        detect = Detect()
        detect.image = check_chess(image)
        detect.thresh()

        # Detect YES/NO
        result = detect.check(detect.crop_tray_1)
        result = np.append(detect.check(detect.crop_tray_2))
        result = np.append(detect.check(detect.crop_tray_3))
        result = np.append(detect.check(detect.crop_tray_4))
        return(result)
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    
    print("123")

    # path = "Camera_test/w7/w7 (3).jpg"
    # image = cv2.imread(path)

   
    # ex.update_check_image(image)

    
    # ex.init_statistic()

    # file_test = cv2.imread('Camera_test/tray/tray (14).jpg') #demo
    # img = check_chess(file_test)
    # # cv2.imwrite('anh.png', img)
    # # cv2.imshow("hâhaaha", img)
    # detect = Detect()
    # detect.image = img
    # # detect.image = cv2.resize(detect.image, (1920, 1080), interpolation=cv2.INTER_AREA)

    # # detect.get_coord()
    # detect.thresh()
    # mask = detect.check(detect.crop_tray_1)
    # mask = np.append(mask, detect.check(detect.crop_tray_2))
    # mask = np.append(mask, detect.check(detect.crop_tray_3))
    # mask = np.append(mask, detect.check(detect.crop_tray_4))
    # print(mask)

    result = np.zeros(192, dtype=int)
    check_yes = np.array([0,1,2,3,4, 7, 46, 47, 95, 96, 97, 98, 143, 144, 190,191])
                
    for i in range(192):
        for j in range(check_yes.size):
            if i == check_yes[j]: 
                result[i] = 1
    print(result)
    ex.update_YesNo_data_to_table(result)
    # ex.update_data(mask)
    sys.exit(app.exec_())

    