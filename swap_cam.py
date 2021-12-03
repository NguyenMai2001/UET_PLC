import cv2
from matplotlib import pyplot as plt

def check_mean(mean):

    image = cv2.resize(mean,(1920,1080))
    height, weight = 1080, 1920
    start_row,start_col= int((height/2)),int((weight/2))
    end_row,end_col= int((height/2)*2),int((weight/2)*2) 
    cropped=image[start_row:end_row,start_col:end_col]
    gray_tray_1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_tray_1, 200, 255, cv2.THRESH_BINARY)

    # cv2.imshow("ing", cropped)
    # cv2.waitKey(10000)

    histr1 = cv2.calcHist([thresh1], [0], None, [256], [0, 256])

    value = int((min(histr1) + max(histr1))/2)
    # print(value)

    # plt.subplot(121)
    # plt.imshow(thresh1)
    # plt.subplot(122)
    # plt.plot(histr1)
    # plt.show()
    return value


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


val_mean_detectImage = check_mean(image)
val_mean_checkImage = check_mean(image1)

print("detect", val_mean_detectImage, ", check", val_mean_checkImage)


# file_name = "swap_cam_check.jpg"
# img=cv2.imread(file_name)
# val_check = check_mean(img)
# print("val_check: " , val_check)

# file_name1 = "tray_check (1).jpg"
# img1=cv2.imread(file_name1)
# val_tray = check_mean(img1)
# print("val_tray1: " , val_tray)

# file_name2 = "tray_check (2).jpg"
# img2=cv2.imread(file_name2)
# val_tray2 = check_mean(img2)
# print("val_tray2: " , val_tray2)

# file_name3 = "tray_check (3).jpg"
# img3=cv2.imread(file_name3)
# val_tray3 = check(img3)
# print("val_tray3: " , val_tray3)

# gt = 0
# if val_check > 222222:
#     gt = 1