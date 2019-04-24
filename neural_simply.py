import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('dataset.csv').values
N,d = data.shape
x = data[:,0:d-1].reshape(-1,2)
y = data[:,d-1].reshape(-1,1)

x = np.hstack((np.ones((N,1)),x))
#khai bao bien

w = np.array([0., 0.5, 1.]).reshape(-1,3)


"""
2-D transposed convolutions (N = 2),
square inputs (i_1 = i_2 = i),
square kernel size (k_1 = k_2 = k),
same strides along both axes (s_1 = s_2 = s),
same zero padding along both axes (p_1 = p_2 = p).
"""
class neural_net():
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate
        self.W = []
        self.b = []
        self.padding = 0
        self.stride = 1

    def sigmoid(self, Z):
        return 1/(1 + np.exp(Z))

    def softmax(self, Z):
        e_Z = np.exp(Z)
        return e_Z / e_Z.sum(axis=0)

    def softmax_stable(self,Z):
        c = abs(np.max(Z, axis=0))
        return np.exp(Z-c) / np.exp(Z-c).sum(axis=0)
    def relu(self, Z):
        return np.max(Z,0)

    def cross_entropy_sofmax(self,Y,Z):
        return -np.sum(np.multiply(Y, np.log(Z)))

    def feedforward(self, X):
        for i in range(self.layers -1):
            Z = np.sum(np.dot(X, self.W), self.b)
            O = self.sigmoid(Z)
            if i < self.layers - 1:
                self.feedforward(O)
        return O
    def backpropragation(self, X, y):
        Out = self.feedforward(X)
        db = []
        dW = []
        dA = []
        i = self.layers - 1
        while i >= 0:
            self.b.append(Out - y)
            self.w.append(np.multiply(Out - y, Out))





