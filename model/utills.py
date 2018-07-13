import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def sigmoid_backward(dA, cache):
    A = sigmoid(cache)
    return dA*A*(1-A)
def relu(Z):
    return np.max(0, Z)
def log(msg):
    return
def relu_backward(A, cache):
    dZ = np.array(dA, copy=True)
    dZ[cache < 0.5] = 0
    return dZ
def linear_backward(A, cache):
    return 1,1,1
def dict2vec(param):
    return