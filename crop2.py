import cv2
from matplotlib import pyplot as plt


img=cv2.imread("file_camimg/d1.jpg")
img_w=cv2.imread("file_camimg/l6.jpg")

# scale_percent = 50

# #calculate the 50 percent of original dimensions
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)

# # dsize
# dsize = (width, height)

# img = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
cv2.imshow("Original Image",img)
# print(img.shape)
cv2.imshow("Original Image lech",img_w)

height,width=img.shape[:2]
start_row,start_col= 410,783
end_row,end_col= 635,1170

# height1,width1=img_w.shape[:2]
# start_row1,start_col1= 20,20
# end_row1,end_col1= 70,108

cropped=img[start_row:end_row,start_col:end_col]
gray_tray_1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
# ret_tray_1, crop_tray_1 = cv2.threshold(gray_tray_1, 0, 255, cv2.THRESH_OTSU)

cropped1=img_w[start_row:end_row,start_col:end_col]
gray_tray_2 = cv2.cvtColor(cropped1, cv2.COLOR_BGR2GRAY)
# ret_tray_2, crop_tray_2 = cv2.threshold(gray_tray_1, 0, 255, cv2.THRESH_OTSU)

histr1 = cv2.calcHist([gray_tray_1], [0], None, [256], [0, 256])
histr2 = cv2.calcHist([gray_tray_2], [0], None, [256], [0, 256])

# print(histr2)
# print(histr2)

plt.subplot(121)
plt.imshow(gray_tray_2)
plt.subplot(122)
plt.plot(histr2)
plt.show()
if max(histr2) > 20000:
    print("true") #co lech
else:
    print("false")

cv2.imshow("Cropped_Image",gray_tray_1)
cv2.imshow("Cropped_w",gray_tray_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
