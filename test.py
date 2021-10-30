import cv2
from matplotlib import pyplot as plt

def check(mean):
    # img=cv2.imread("file_camimg/d1.jpg")
    # img_w=cv2.imread("file_camimg/l1") #demo
    image = cv2.resize(mean,(1920,1080))
    start_row,start_col= 463,795
    end_row,end_col= 637,1135

    cropped=image[start_row:end_row,start_col:end_col]
    gray_tray_1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # ret_tray_1, crop_tray_1 = cv2.threshold(gray_tray_1, 0, 255, cv2.THRESH_OTSU)

    histr1 = cv2.calcHist([gray_tray_1], [0], None, [256], [0, 256])

    plt.subplot(121)
    plt.imshow(gray_tray_1)
    plt.subplot(122)
    plt.plot(histr1)
    plt.show()

    return histr1

# img=cv2.imread("Camera_test/w2/w2 (28).jpg")
img=cv2.imread("Camera_test/correct1/d (7).jpg")

value = check(img)

if max(value) > 10000 or max(value) < 9500:
    print("true") #co lech
else:
    print("false") #không lệch
