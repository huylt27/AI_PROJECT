import pandas as pd
import numpy as np

def conv2D(X, K):
    w, h = K.shape
    Y = np.zeros((X.shape[0] - w + 1, X.shape[1] - h + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+w, j:j+h] * K).sum()
    return Y
x = np.arange(0,9).reshape(3,3)
k = np.arange(0,4).reshape(2,2)

def softmax(Z):
    e_Z = np.exp(Z)
    return e_Z / e_Z.sum(axis = 1, keepdims=True)

def softmax_stable(Z):
    c = np.max(Z, axis = 0)
    e_Z = np.exp(np.absolute(Z) - c)
    return e_Z / e_Z.sum(axis=1, keepdims=True)

#define the loss function with cross-entropy
def softmax_cross_entropy(Z, y):
    y_pre = softmax(Z)
    loss_cross_entropy = -np.multiply(y, y_pre).sum(axis = 0, keepdims=True)
    return loss_cross_entropy


def relu(Z):
    return np.max(Z,0)
