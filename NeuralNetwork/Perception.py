import numpy as np

class Perception:
    def __init__(self, N, learning_rate):
        self.W = np.random.randn((N + 1) / np.sqrt(N))
        self.learning_rate = learning_rate
    #define a steps function to compare the loss
    def steps(self,x):
        return 1 if x > 0 else 0

    #define the fit function to train

    def fit(self, X, y, epoches):


        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in range(epoches):
            # tính toán đầu ra dự đoán và so sánh với giá trị thực qua hàm steps
            for (x,target) in zip(X,y):
                p = self.steps(np.dot(X, self.W))
                if p != target:
                    loss = p - target
                    self.W += -self.learning_rate*loss.x
    #định nghĩa hàm dự đoán kết quả
    def predict(self, X, bias=True):
        #kiểm tra đầu vào X có phải là 1 ma trận hay không
        X = np.atleast_2d(X)

        #kiểm tra đã thêm bias vào ma trận x hay chưa
        if bias:
             X = np.c_[X, np.ones((X.shape[0]))]
        return self.steps(np.dot(X, self.W))
