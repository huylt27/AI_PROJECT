import numpy as np
from os import sys
import cv2

class layers:
    def __init__(self, node, kernel_size):
        self.node = node
        self.kernel_size = kernel_size

    #define the Conv2D with default stride = 1 and have one the output
    def Conv2D(self, x, padding = (0,0)):
        h,w = self.kernel_size.shape
        if len(x.shape) == 2 or len(self.kernel_size.shape) == 2:
            if x.shape[-1] != self.kernel_size.shape[-1]:
                print("Error: The channel input {} and channle kernel {} not be match".format(x.shape[-1]), self.kernel_size.shape[-1])
                sys.exit()
        if self.kernel_size.shape[0] != self.kernel_size.shape[1]:
            print("Recommend the kernel size should be match size: ie: height and width must be match")
            sys.exit()
        if self.kernel_size.shape[0] % 2 ==0:
            print("Recommend the kernel size should be the odd number.")
            sys.exit()
        p_h, p_w = padding
        y = np.zeros((x.shape[0] - h + 1 + p_h, x.shape[1] - w + 1 + p_w))

        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i,j] = (x[i:i+h, j:j+w]).sum()
        return y

    def Conv3D(self, x):
        h, w, c = self.kernel_size.shape
        if len(x.shape) == len(self.kernel_size.shape) == 3:
            if x.shape[-1] != self.kernel_size.shape[-1]:
                print("Error: The channel input {} and channle kernel {} not be match".format(x.shape[-1]),
                      self.kernel_size.shape[-1])
                sys.exit()
            if self.kernel_size.shape[0] != self.kernel_size.shape[1]:
                print("Recommend the kernel size should be match size: ie: height and width must be match")
                sys.exit()
            if self.kernel_size.shape[0] % 2 == 0:
                print("Recommend the kernel size should be the odd number.")
                sys.exit()
            for i in range(c):
                return
