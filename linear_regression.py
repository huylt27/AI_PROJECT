import numpy as np
import  pandas as pd

#fake dữ liệu training
x_train = np.array([[20,1],[24,1],[18,1],[16,1],[30,1],[28,1],[50,2],[60,2],[55,2],[60,2],[65,2],[70,2],
                      [85,3],[90,3],[80,3],[101,3],[62,2],[75,3],[40,2],[58,2]], dtype='float32')

y_train = np.array([2000000,2500000,2300000,1800000,3000000,2800000,4000000,4600000,4300000,5000000,5400000,
                    6000000,7500000,8000000,7800000,9000000,4800000,6800000,3800000,5600000], dtype='float32')
x_train = x_train.reshape(-1,2)
y_train = y_train.reshape(-1,1)

N = x_train.shape[0]
b = np.linspace(0., 1.,20).reshape(-1,1)
w = np.array([3, 4], dtype='float32').reshape(-1,1)
learning_rate = 0.00001
y_pre = np.dot(x_train,w) + b
for i in range(20):
    #tính r = y^-y
    r = y_train - y_pre
    loss = 0.5*sum(r*r)/N

    #cập nhật giá trị w
    w[0] -= learning_rate*sum(np.multiply(x_train[:,0], r))
    w[1] -= learning_rate*sum(np.multiply(x_train[:,1], r))
    print(loss)