import time

import numpy as np


class Softmax:
    def __init__(self):
        pass

    @staticmethod
    def __softmax_function(z):
        ez = np.exp(z)
        ez_sum = np.sum(ez, axis=1).reshape((-1, 1))
        # print(ez)
        # print(ez_sum)
        return np.divide(ez, ez_sum)

    def cost(self, X, y, W, b):
        # Weights are in columns as w1 in column 1, w2 in column 2 and so on.
        z = np.matmul(X, W) + b
        # print(z)
        # Stable Softmax (prevents overflow)
        z = z - np.max(z, axis=1).reshape((X.shape[0], 1))
        # z = np.arange(9).reshape((-1,3))
        # print(z)
        f_wb = self.__softmax_function(z)
        # y = np.array([1,2,0])
        # t = time.time()
        y_hat = f_wb[np.arange(y.shape[0]),y]
        cost = np.sum(-np.log(y_hat))
        # print('t1', time.time() - t)
        # t = time.time()
        # cost2 = 0
        # for i,val in enumerate(y):
        #     f_wb_i = f_wb[i][val]
        #     cost2 += -np.log(f_wb_i)
        # print('t2', time.time() - t)
        # print(cost, cost2)
        return cost

    def __gradient(self):
        pass