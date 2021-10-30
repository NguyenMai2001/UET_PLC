import cv2
from matplotlib import pyplot as plt

def check(mean):

    image = cv2.resize(mean,(1920,1080))
    start_row,start_col= 479,689
    end_row,end_col= 697,1061
    cropped=image[start_row:end_row,start_col:end_col]
    gray_tray_1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_tray_1, 120, 255, cv2.THRESH_BINARY)

    # cv2.imshow("ing", cropped)
    # cv2.waitKey(10000)

    histr1 = cv2.calcHist([thresh1], [0], None, [256], [0, 256])

    value = int((min(histr1)+max(histr1))/2)
    # print(value)

    # plt.subplot(121)
    # plt.imshow(thresh1)
    # plt.subplot(122)
    # plt.plot(histr1)
    # plt.show()
    return value

for alpha in range(5,500,5):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    TPR = 0
    FPR = 0
    for i in range(100):
        file_name = "Camera_test/w6/w6 (" + str(i+1) + ").jpg"
        img=cv2.imread(file_name)
        # img=cv2.imread("Camera_test/cr3/cr3 (7).jpg")
        # img=cv2.imread("file_camimg/d1.jpg") #demo


        value_in = 38080
        value_out = check(img)

        if value_out < (value_in - alpha) or value_out > (value_in + alpha) :
            # print("true") #co lech
            # print("annh thu ", i) #không lệch
            # print("gia tri cuong do: ", value_out) #có lệch
            TP = TP + 1
            # tb = tb + value_out
        else:
            # print("annh thu ", i+1)
            # print("gia tri cuong do: ", value_out) #không lệch
            FN = FN + 1

    for i in range(100):
        file_name = "Camera_test/cr6/cr6 (" + str(i+1) + ").jpg"
        img=cv2.imread(file_name)
        # img=cv2.imread("Camera_test/cr3/cr3 (7).jpg")
        # img=cv2.imread("file_camimg/d1.jpg") #demo


        value_in = 38080
        value_out = check(img)

        if value_out < (value_in - alpha) or value_out > (value_in + alpha) :
            # print("true") #co lech
            # print("annh thu ", i) #không lệch
            # print("gia tri cuong do: ", value_out) #có lệch
            FP = FP + 1
            # tb = tb + value_out
        else:
            # print("annh thu ", i+1)
            # print("gia tri cuong do: ", value_out) #không lệch
            TN = TN + 1
            
    
    print('===========================================')
    print('TP ' + str(alpha) + ': ', TP)
    print('FN ' + str(alpha) + ': ', FN)
    print('FP ' + str(alpha) + ': ', FP)
    print('TN ' + str(alpha) + ': ', TN)

    TPR = (TP/ (TP+FN))*100
    FPR = (FP/ (FP + TN))*100
    print('TPR = ', TPR)
    print('FPR = ', FPR)


    # print("trung binh gt diem anh: ", tb/100)


