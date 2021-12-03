#file lien quan train_data_load.py, log_regress.py
import cv2
import pickle
from matplotlib import pyplot as plt

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

height, weight = 1080, 1920
start_row,start_col= int((height/10)),int((weight/10)*4)
end_row,end_col= int((height/10)*4),int((weight/10)*6) 
count = 0
# plt.subplots(3,4,figsize=(23,30))
num = 100
for i in range(num):
    path = "Camera_test/cr7/cr7 (" + str(i+1) + ").jpg"
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_img = img[start_row:end_row,start_col:end_col]

    sample_img = new_img.reshape(-1,int((height/10)*3)*int((weight/10)*2))
    ypred = clf.predict(sample_img)
    if ypred[0] == 1:
        count = count +1
    # print(ypred)
    # print(i)
    # print('=========')

    # plt.subplot(3,4,i+1)
    # plt.imshow(new_img.reshape((360,640)), cmap='gray')
    # plt.title(f'ypred={ypred[0]}')

# plt.show()
print(count)
print("Accuracy: ", (count/num)*100 , "%")

