from re import S
import sklearn
import pickle
import cv2
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



pickle_in = open('X.pickel', 'rb')
X = pickle.load(pickle_in)

pickle_in = open('y.pickel', 'rb')
y = pickle.load(pickle_in)

xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=101)
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


n = 162
d = 230400
height, weight = 1080, 1920

def draw_sample_label(X,y,ypred=None):
    X = X[:12]
    y = y[:12]
    plt.subplots(3,4,figsize=(23,37))
    for i in range(len(X)):
        plt.subplot(3,4,i+1)
        plt.imshow(X[i].reshape(int((height/10)*3),int((weight/10)*2)), cmap='gray')
        if ypred is None:
            plt.title(f'y={y[i]}')
        else:
            plt.title(f'y={y[i]} ypred={ypred[i]}')
    plt.show()

draw_sample_label(X,y)

clf = LogisticRegression(max_iter=10000)
clf.fit(X,y)

# ypred = clf.predict(xte)
# print(ypred)
#print(f"error rate {(yte!=ypred).sum() / len(yte)*100:2f}%")
# mask = yte != ypred
# draw_sample_label(xte[mask], yte[mask], ypred[mask])

#luu clf
with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)


