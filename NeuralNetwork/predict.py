from NeuralNetwork.Perception import Perception
import numpy as np
#fake dữ liệu
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

#khởi tạo giá trị cho class perception
p = Perception(X.shape[1], learning_rate =0.01)

#bắt đầu training
p.fit(X, y, epoches=20)

for (x, labels) in zip(X,y):
    pred = p.predict(x)
    print("[INFO]: Data {} -- labels {} -- pred {}".format(x, labels, pred))