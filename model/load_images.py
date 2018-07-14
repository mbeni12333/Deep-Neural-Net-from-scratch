import scipy.misc as sp
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import sys

def cut_square(img):
    return

X=np.empty((12288,1))
Y=[]
path = 'dataset/'
classes = {}

for j, dirr in enumerate(os.listdir(path)):
    fig = plt.figure(figsize=(8, 8))
    classes[j] = dirr
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i, f in enumerate(os.listdir(path+dirr)):
        print(f"file = {os.path.join(dirr, f)}")
        img = np.array(sp.imread(path+dirr+'/'+f))
        print(f"image : {f} is of shape : {img.shape}")
        img = sp.imresize(img, (64,64))
        ax = fig.add_subplot(5,5, i+1 ,xticks=[], yticks=[])
        ax.imshow(img)
        img = img.reshape(64*64*3, 1)
        print(f"image : {f} reshaped to : {img.shape}")
        X = np.concatenate((X,img), axis=1)
        Y.append(j)
Y = np.array(Y).reshape(1, len(Y))
print(f"X is a shape of {X.shape}")
print(f"trainning examples : {X.shape[1]}")
permutation = np.random.permutation(50)
X_shuffled = X[:, permutation]
Y_shuffled = Y[:, permutation]
fig2 = plt.figure(figsize=(8,8))
for i in range(1,26):
    img2 = X_shuffled[:,i].reshape(64,64,3) - np.mean(X)
    img2 /= np.std(X)
    ax2 = fig2.add_subplot(5, 5, i, xticks=[], yticks=[])
    ax2.set_title(classes[int(Y_shuffled[:, i])])
    ax2.imshow(img2)

plt.show()
