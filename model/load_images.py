import scipy.misc as sp
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import sys
plt.style.use("ggplot")
def cut_square(img):
    return
def to_one_hot(y):
    return
def load_fake_data(dataset):
    fig = plt.figure(figsize=(6,6))
    x,y = sk.make_regression( n_features=2, n_samples=500, noise=0.3)
    x = x*7
    x[:,1] += x[:,0]*2 + 6
    a1 = fig.add_subplot(1,1,1)

    a1.scatter(x[:,0], x[:,1], c=y, s=50)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.axis('equal')

    fig.suptitle("diabetes")
    a1.set_xlabel("x component")
    a1.set_ylabel("y component")

    ### normalizing
    print(f"mean = {np.mean(x, axis=0)} , standard deviation = {np.std(x, axis=0)}")
    x = x - np.mean(x, axis=0)
    x = x / np.std(x, axis=0)
    cov = np.dot(x.T, x) / x.shape[0]
    U, S, V = np.linalg.svd(cov)
    x_rot = np.dot(x, U)
    x = x_rot / np.sqrt(S + 1e-10)
    fig2 = plt.figure(figsize=(6, 6))
    a2 = fig2.add_subplot(1, 1, 1)
    fig2.suptitle("diabetes normalized")
    a2.scatter(x[:,0], x[:,1], c=y, s=50)
    a2.set_xlabel("x component")
    a2.set_ylabel("y component")
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.axis('equal')

    plt.show()
    return
def load_image_dataset(path):
    X=np.empty((12288,1))
    Y=[]
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
        img2 = ((X_shuffled[:,i]- np.mean(X_shuffled, axis=1))/np.std(X_shuffled, axis=1)).reshape(64,64,3)
        ax2 = fig2.add_subplot(5, 5, i, xticks=[], yticks=[])
        ax2.set_title(classes[int(Y_shuffled[:, i])])
        ax2.imshow(img2)

    plt.show()

if __name__ == '__main__':
    load_fake_data(None)