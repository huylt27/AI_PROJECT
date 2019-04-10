import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#get the data input

data = pd.read_csv('datashets.csv').values
#so input
N = data.shape[0]
#tach x, y
x_train = data[:, [0,1]].reshape(-1,2)

y_train = data[:,2].reshape(-1,1)

b = np.linspace(0,1,19).reshape(-1,1)
#giá trị khởi tạo w
w = np.array([0.,1.]).reshape(-1,1)
#x_train = np.hstack((np.ones((N,1)),x_train))
learning_rate = 0.0000005
for i in range(100):
    #tinh y^-y
    r = np.add(np.dot(x_train,w),b) - y_train
    #tinh loss
    loss = 0.5*np.sum(r*r)/N
    b[:] -= learning_rate*np.sum(r)
    w[0] -= learning_rate*np.sum(np.multiply(r, x_train[:,0]))
    w[1] -= learning_rate*np.sum(np.multiply(r, x_train[:,1]))
    print(loss)
predict = np.add(np.dot(x_train,w),b)
# plt.plot(x_train[0], x_train[N-1], predict[0], predict[N-1], 'r')
# plt.show()

x1 = 49
x2 = 2
y = b[0] + w[0]*x1 + w[1]*x2
print('gia du  doan', y)