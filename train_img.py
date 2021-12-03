import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import glob
import sklearn
import pickle
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from numpy.core.fromnumeric import resize

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def check_chess(file_test): # file_test = cv2.imread("")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('*.jpg')
    for i in range(2):
        file_name = "images/chess (" + str(i+1) + ").jpg"
        img = cv2.imread(file_name)
        # img = cv.resize(img1, (640,480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            # cv.imshow('img', img)
            # print(i+1)
            # cv.waitKey(500)
    # cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # file_test = 'tray/tray (14).jpg'
    # img = cv2.imread(file_test)
    # img = cv.resize(img1, (640,480))
    img = file_test
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

class Detect(object):
    def __init__(self) -> None:
        super().__init__()

        self.image = []
        self.crop_tray_1 = []
        self.crop_tray_2 = []
        self.crop_tray_3 = []
        self.crop_tray_4 = []

        f = open(resource_path('data/config/location_crop_yn.txt'))
        self.t1_x_begin = int(f.readline()) # 202
        self.t1_y_begin = int(f.readline()) #321   
        self.t1_x_end = int(f.readline()) #429
        self.t1_y_end = int(f.readline())  #515    
        self.t2_x_begin = int(f.readline())
        self.t2_y_begin = int(f.readline())
        self.t2_x_end = int(f.readline())
        self.t2_y_end = int(f.readline())
        self.t3_x_begin = int(f.readline())
        self.t3_y_begin = int(f.readline())
        self.t3_x_end = int(f.readline())
        self.t3_y_end = int(f.readline())
        self.t4_x_begin = int(f.readline())
        self.t4_y_begin = int(f.readline())
        self.t4_x_end = int(f.readline())
        self.t4_y_end = int(f.readline())

    def find_location_crop(self, event, x, y, flags, param):
        f = open(resource_path('data/config/location_crop_yn.txt'), 'w')
        if event == cv2.EVENT_LBUTTONDOWN:
            f.write(str(x) + "\n")
            f.write(str(y) + "\n")
    
    def get_coord(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.find_location_crop)
        while True:
            cv2.imshow("image", self.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def thresh(self):
        
        self.crop_tray_1 = self.image[self.t1_y_begin:self.t1_y_end, self.t1_x_begin:self.t1_x_end]
        self.crop_tray_2 = self.image[self.t2_y_begin:self.t2_y_end, self.t2_x_begin:self.t2_x_end]
        self.crop_tray_3 = self.image[self.t3_y_begin:self.t3_y_end, self.t3_x_begin:self.t3_x_end]
        self.crop_tray_4 = self.image[self.t4_y_begin:self.t4_y_end, self.t4_x_begin:self.t4_x_end]

        # gray_tray_1 = cv2.cvtColor(crop_tray_1, cv2.COLOR_BGR2GRAY)
        # gray_tray_2 = cv2.cvtColor(crop_tray_2, cv2.COLOR_BGR2GRAY)
        # gray_tray_3 = cv2.cvtColor(crop_tray_3, cv2.COLOR_BGR2GRAY)
        # gray_tray_4 = cv2.cvtColor(crop_tray_4, cv2.COLOR_BGR2GRAY)

        # ret_tray_1, self.crop_tray_1 = cv2.threshold(gray_tray_1, 120, 255, cv2.THRESH_BINARY)
        # ret_tray_2, self.crop_tray_2 = cv2.threshold(gray_tray_2, 120, 255, cv2.THRESH_BINARY)
        # ret_tray_3, self.crop_tray_3 = cv2.threshold(gray_tray_3, 120, 255, cv2.THRESH_BINARY)
        # ret_tray_4, self.crop_tray_4 = cv2.threshold(gray_tray_4, 120, 255, cv2.THRESH_BINARY)

    # def check(self, crop_img):
    #     height = crop_img.shape[0]
    #     width = crop_img.shape[1]
    #     mask = np.zeros(48)
    #     for i in range(48):
    #         k = int(i / 6)
    #         j = i % 6
    #         cut = crop_img[int(height / 6 * (6 - j - 1)):int(height / 6 * (6 - j)), int(width / 8 * k):int(width / 8 *
    #                                                                                                         (k + 1))]
    #         histr = cv2.calcHist([cut], [0], None, [256], [0, 256])
    #         # plt.subplot(121)
    #         # plt.imshow(cut)
    #         # plt.subplot(122)
    #         # plt.plot(histr)
    #         # plt.show()
    #         # print(i+1)
    #         if histr[255] >= 800:
    #             mask[i] = 1
    #     return mask

    def rotated(self, image):
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=1, scale=1)
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        cv2.imshow('Rotated image', rotated_image)
        cv2.waitKey(0)
        # cv2.imwrite('rotated_image.png', rotated_image)
        return rotated_image

classes = ['no', 'yes']
train_tray = []
X = []
y = []

def add_crop_yes(crop_img):
    cv2.imwrite('img_y.jpg', crop_img)
    img = cv2.imread('img_y.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.show()

    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros(48)
    for i in range(48):
        k = int(i / 6)
        j = i % 6
        cut = img[int(height / 6 * (6 - j - 1)):int(height / 6 * (6 - j)), int(width / 8 * k):int(width / 8 *(k + 1))]
        cut1 = cv2.resize(cut,(31,27))
        train_tray.append([cut1, classes.index('yes')])
        # plt.imshow(cut, cmap='gray')
        # plt.show()

def add_crop_no(crop_img):
    cv2.imwrite('img_n.jpg', crop_img)
    img = cv2.imread('img_n.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.show()

    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros(48)
    for i in range(48):
        k = int(i / 6)
        j = i % 6
        cut = img[int(height / 6 * (6 - j - 1)):int(height / 6 * (6 - j)), int(width / 8 * k):int(width / 8 *(k + 1))]
        cut1 = cv2.resize(cut,(31,27))
        train_tray.append([cut1, classes.index('no')])
        # plt.imshow(cut, cmap='gray')
        # plt.show()

if __name__ == "__main__":
#     #=========================DATA LAN 1 ==========================================================
#     file_test_l1 = cv2.imread('train_tray/train6.jpg') 
#     img_l1 = check_chess(file_test_l1)
#     detect = Detect()
#     detect.image = detect.rotated(img_l1)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_1
#     add_crop_yes(crop_img)

#     #tray ko co cam
#     crop_img = detect.crop_tray_2
#     add_crop_no(crop_img)

#     crop_img = detect.crop_tray_3
#     add_crop_no(crop_img)

#     crop_img = detect.crop_tray_4
#     add_crop_no(crop_img)

#     #Anh 2
#     file_test2_l1 = cv2.imread('train_tray/train6 (2).jpg') 
#     img2_l1 = check_chess(file_test2_l1)
#     detect = Detect()
#     detect.image = detect.rotated(img2_l1)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_2
#     add_crop_yes(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_1
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_3
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_4
#     add_crop_no(crop_img)

#     #Anh 3
#     file_test3_l1 = cv2.imread('train_tray/train6 (3).jpg') 
#     img3_l1 = check_chess(file_test3_l1)
#     detect = Detect()
#     detect.image = detect.rotated(img3_l1)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_3
#     add_crop_yes(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_1
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_2
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_4
#     add_crop_no(crop_img)

#     #Anh 4
#     file_test4_l1 = cv2.imread('train_tray/train6 (4).jpg') 
#     img4_l1 = check_chess(file_test4_l1)
#     detect = Detect()
#     detect.image = detect.rotated(img4_l1)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_4
#     add_crop_yes(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_1
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_2
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_3
#     add_crop_no(crop_img)

#     #=========================DATA LAN 2 ==========================================================
#     file_test_l4 = cv2.imread('train_tray/train5.jpg') 
#     img_l4 = check_chess(file_test_l4)
#     detect = Detect()
#     detect.image = detect.rotated(img_l4)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_1
#     add_crop_yes(crop_img)

#     #tray ko co cam
#     crop_img = detect.crop_tray_2
#     add_crop_no(crop_img)

#     crop_img = detect.crop_tray_3
#     add_crop_no(crop_img)

#     crop_img = detect.crop_tray_4
#     add_crop_no(crop_img)

#     #Anh 2
#     file_test2_l4 = cv2.imread('train_tray/train5 (2).jpg') 
#     img2_l4 = check_chess(file_test2_l4)
#     detect = Detect()
#     detect.image = detect.rotated(img2_l4)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_2
#     add_crop_yes(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_1
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_3
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_4
#     add_crop_no(crop_img)

#     #Anh 3
#     file_test3_l4 = cv2.imread('train_tray/train5 (3).jpg') 
#     img3_l4 = check_chess(file_test3_l4)
#     detect = Detect()
#     detect.image = detect.rotated(img3_l4)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_3
#     add_crop_yes(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_1
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_2
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_4
#     add_crop_no(crop_img)

#     #Anh 4
#     file_test4_l4 = cv2.imread('train_tray/train5 (4).jpg') 
#     img4_l4 = check_chess(file_test4_l4)
#     detect = Detect()
#     detect.image = detect.rotated(img4_l4)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_4
#     add_crop_yes(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_1
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_2
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_3
#     add_crop_no(crop_img)

# #=========================DATA LAN 3 ==========================================================
#     file_test_2tray = cv2.imread('train_tray/train1_4.jpg') 
#     img_2tray = check_chess(file_test_2tray)
#     detect = Detect()
#     detect.image = detect.rotated(img_2tray)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_1
#     add_crop_yes(crop_img)

#     #tray co cam
#     crop_img = detect.crop_tray_4
#     add_crop_yes(crop_img)

#     #tray ko co cam
#     crop_img = detect.crop_tray_2
#     add_crop_no(crop_img)

#     crop_img = detect.crop_tray_3
#     add_crop_no(crop_img)


#     #Anh 2
#     file_test_2tray2 = cv2.imread('train_tray/train2_3.jpg') 
#     img_2tray2 = check_chess(file_test_2tray2)
#     detect = Detect()
#     detect.image = detect.rotated(img_2tray2)
#     detect.thresh()

#     #tray co cam
#     crop_img = detect.crop_tray_2
#     add_crop_yes(crop_img)

#     #tray co cam
#     crop_img = detect.crop_tray_3
#     add_crop_yes(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_1
#     add_crop_no(crop_img)

#     #tray khong cam
#     crop_img = detect.crop_tray_4
#     add_crop_no(crop_img)

#=========================DATA LAN 3 ==========================================================
    file_test_2tray = cv2.imread('train_tray/train1_2.jpg') 
    img_2tray = check_chess(file_test_2tray)
    detect = Detect()
    detect.image = detect.rotated(img_2tray)
    detect.thresh()

    #tray co cam
    crop_img = detect.crop_tray_1
    add_crop_yes(crop_img)

    #tray co cam
    crop_img = detect.crop_tray_2
    add_crop_yes(crop_img)

    #tray ko co cam
    crop_img = detect.crop_tray_3
    add_crop_no(crop_img)

    crop_img = detect.crop_tray_4
    add_crop_no(crop_img)


    #Anh 2
    file_test_2tray2 = cv2.imread('train_tray/train3_4.jpg') 
    img_2tray2 = check_chess(file_test_2tray2)
    detect = Detect()
    detect.image = detect.rotated(img_2tray2)
    detect.thresh()

    #tray co cam
    crop_img = detect.crop_tray_3
    add_crop_yes(crop_img)

    #tray co cam
    crop_img = detect.crop_tray_4
    add_crop_yes(crop_img)

    #tray khong cam
    crop_img = detect.crop_tray_1
    add_crop_no(crop_img)

    #tray khong cam
    crop_img = detect.crop_tray_2
    add_crop_no(crop_img)

    # =============================================================================
    
    print(np.shape(train_tray))
    import random
    random.shuffle(train_tray)

    X=[] #tap anh
    y=[] #tap nhan

    for img, label in train_tray:
        X.append(img)
        y.append(label)

    # print(np.shape(X[-1]))
    # print(type(X))
    # print(np.shape(X[-1]))

    X = np.array(X).reshape(-1,31*27)
    y = np.array(y)

    #TRAIN DATA

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.05, random_state=101)
    # print(X,y)
    # print(X.shape)
    # print(y.shape)
    # print(xtr)
    # print(xtr.shape)
    # print(xte.shape)
    # print(ytr.shape)
    # print(yte.shape)

    # plt.imshow(X[1].reshape((360,640)), cmap='gray')
    # plt.show()


    n = 384
    d = 837

    def draw_sample_label(X,y,ypred=None):
        X = X[:12]
        y = y[:12]
        plt.subplots(3,4)
        for i in range(len(X)):
            plt.subplot(3,4,i+1)
            plt.imshow(X[i].reshape(27,31), cmap='gray')
            if ypred is None:
                plt.title(f'y={y[i]}')
            else:
                plt.title(f'y={y[i]} ypred={ypred[i]}')
        plt.show()

    draw_sample_label(X,y)

    clf_tray = LogisticRegression(max_iter=10000)
    clf_tray.fit(X,y)

    # ypred = clf_tray.predict(xte)
    # print(ypred)
    # print(f"error rate {(yte!=ypred).sum() / len(yte)*100:2f}%")
    # mask = yte != ypred
    # draw_sample_label(xte[mask], yte[mask], ypred[mask])

    #luu clf
    with open('clf_tray.pkl', 'wb') as f:
        pickle.dump(clf_tray, f)  