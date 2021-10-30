import cv2
from matplotlib import pyplot as plt

def check(mean):

    image = cv2.resize(mean,(1920,1080))
    start_row,start_col= 479,689
    end_row,end_col= 697,1061
    cropped=image[start_row:end_row,start_col:end_col]
    gray_tray_1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_tray_1, 120, 255, cv2.THRESH_BINARY)

    # cv2.imshow("ing", thresh1)
    # cv2.waitKey(0)

    histr1 = cv2.calcHist([thresh1], [0], None, [256], [0, 256])

    value = int((min(histr1)+max(histr1))/2)
    # print(value)

    # plt.subplot(121)
    # plt.imshow(thresh1)
    # plt.subplot(122)
    # plt.plot(histr1)
    # plt.show()
    return value

count_l = 0
count_cr = 0
tb = 0
for i in range(100):
    file_name = "Camera_test/w6/w6 (" + str(i+1) + ").jpg"
    img=cv2.imread(file_name)
    # img=cv2.imread("Camera_test/cr3/cr3 (7).jpg")
    # img=cv2.imread("file_camimg/d1.jpg") #demo


    value_in = 38083
    value_out = check(img)

    if value_out < (value_in - 500) or value_out > (value_in + 500) :
        # print("true") #co lech
        # print("annh thu ", i) #không lệch
        # print("gia tri cuong do: ", value_out) #có lệch
        count_l = count_l + 1
        tb = tb + value_out
    else:
        print("annh thu ", i+1)
        # print("gia tri cuong do: ", value_out) #không lệch
        count_cr = count_cr + 1

print('co lech: ', count_l)
print('khong lech: ', count_cr)
print("trung binh gt diem anh: ", tb/100)


