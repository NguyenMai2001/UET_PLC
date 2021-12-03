import os
import pandas as pd
import random
import cv2

# print(random.random())

# arr=[1, 2, 3]

# cam = cv2.VideoCapture(0)
# cam.set(3,1920)
# cam.set(4,1080)

# ret, img = cam.read()
# cv2.imshow("k", img)
# cv2.waitKey(0)


def save_check(img, arr):
    title = random.random()
    image_path = 'data_check/'
    os.chdir(image_path)
    image_title = str(title)

    name_img = image_title + ".jpg"
    cv2.imwrite(name_img, img)

    name_txt = image_title + ".txt"
    f = open(name_txt,"w+")
    f.write(str(arr))
    f.close()
    print("Write Done")

# save_check(img)


# Yolo_file = Yolo_values.loc[Yolo_values['ImageID'] == image_title]

# datafr = Yolo_file.loc[:, ['classNumber', 'center x', 'center y', 'width', 'height']].copy()

# save_path = image_path + '/' + image_title + '.txt'

# datafr.to_csv(save_path, header=False, index=False, sep=' ')