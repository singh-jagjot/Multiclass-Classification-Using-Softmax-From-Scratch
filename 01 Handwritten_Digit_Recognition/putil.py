import time

import numpy as np


class Softmax:
    def __init__(self):
        self.X = None
        self.y = None
        self.W = None
        self.b = None

    def set_weight(self, W):
        self.W = W

    def set_bias(self, b):
        self.b = b

    def fit(self, X, y):
        self.X = X
        self.y = y

    def fit(self, X, y, W, b):
        self.W = W
        self.b = b
        self.X = X
        self.y = y

    def _softmax(self, z):
        # Stable Softmax (prevents overflow)
        X = self.X
        z = z - np.max(z, axis=1).reshape((X.shape[0], 1))
        ez = np.exp(z)
        ez_sum = np.sum(ez, axis=1).reshape((-1, 1))
        return np.divide(ez, ez_sum)

    def _get_f_wb(self):
        X, W, b = self.X, self.W, self.b
        # Weights are in columns as w1 in column 1, w2 in column 2 and so on.
        z = np.matmul(X, W) + b
        # print(z)
        f_wb = self._softmax(z)
        return f_wb

    def _get_cost(self):
        X, W, b,y = self.X, self.W, self.b,self.y
        # Unfortunately below code won't work because of high possibility of log(0) errors.
        # t = time.time()
        # f_wb = self._get_f_wb()
        # y_hat = f_wb[np.arange(y.shape[0]), y]
        # j_wb = -np.log(y_hat)
        # cost = np.sum(j_wb)
        # print('t1', time.time() - t)

        # Slow code
        # t = time.time()
        # cost2 = 0
        # for i,val in enumerate(y):
        #     f_wb_i = f_wb[i][val]
        #     cost2 += -np.log(f_wb_i)
        # print('t2', time.time() - t)
        # print(cost, cost2)

        # print('cost',cost)
        # Below code works without log(0) errors
        # Weights are in columns as w1 in column 1, w2 in column 2 and so on.
        z = np.matmul(X, W) + b
        z_max = np.max(z, axis=1).reshape((X.shape[0], 1))
        z_new = z-z_max
        ez_new = np.exp(z_new)
        ez_sum = np.sum(ez_new, axis=1).reshape((-1, 1))
        # print('z',z)
        # print('z_max',z_max)
        # print('z_new',z_new)
        # print('ez_new',ez_new)
        # print('ez_sum',ez_sum)
        log_softmax = (z -z_max) - np.log(ez_sum)
        loss = -log_softmax[np.arange(y.shape[0]), y]
        cost = np.sum(loss)
        print('cost', cost)
        return cost

    def _get_gradient(self, alpa=0):
        # 'alpa' is learning rate
        W, b = self.W, self.b
        y = self.y
        dj_dw = self._get_f_wb() - 1
        dj_db = self._get_f_wb() - 1
        print(dj_dw, dj_db)
        return None
