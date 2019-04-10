import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv').values

N, d = data.shape
x = data[:,0:d-1].reshape(-1,2)
y = data[:, d-1].reshape(-1,1)

plt.scatter(x[:10,0], x[:10,1], c='red', edgecolors=None, s=30, label='Cho vay')
plt.scatter(x[10:,0], x[10:, 1], c = 'blue', edgecolors=None, s = 30, label='Khong cho vay')
plt.legend(loc=1)
plt.xlabel('Muc luong')
plt.ylabel('nam kinh nghiem')
plt.show()

x = np.hstack((np.ones((N,1)),x))

w = np.array([0., 0.5, 1.]).reshape(-1,1)
learning_rate = 0.01

def sigmoid(X):
    return 1/np.add(1,np.exp(X))
for i in range(10):
    #tinh y_pre =y^ -y
    y_pre = sigmoid(-np.dot(x,w))
    loss = -np.add(np.multiply(y, np.log(y_pre)), np.multiply((1-y),np.log(1-y_pre)))
    w -= learning_rate*np.dot(x.T, y_pre-y)
    #print(loss)

t = 0.5

x_pre = np.array([1,12,1])
y_pridict =  sigmoid(np.dot(x_pre,w))
if y_pridict > 0.5:
    print('đủ điều kiện cho vay, độ tin cậy: ', y_pridict)
else:
    print('Ko đủ điều kiện vay')



