import time

import numpy as np


class Softmax:
    def __init__(self):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.y_hot: np.ndarray = None
        self.W: np.ndarray = None
        self.b: np.ndarray = None
        self.mu: None
        self.sigma = None

    def set_weight(self, W):
        self.W: np.ndarray = W

    def set_bias(self, b):
        self.b: np.ndarray = b

    def fit(self, X, y):
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self._one_hot_enc()

    def fit(self, X, y, W, b):
        self.W: np.ndarray = W
        self.b: np.ndarray = b
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self._one_hot_enc()

    def get_weights_bias(self):
        '''return -> W, b'''
        return self.W, self.b

    def _softmax(self, z):
        # Stable Softmax (prevents overflow)
        z = z - np.max(z, axis=1).reshape((z.shape[0], 1))
        ez = np.exp(z)
        ez_sum = np.sum(ez, axis=1).reshape((-1, 1))
        return ez / ez_sum

    # def _get_f_wb(self):
    #     X, W, b = self.X, self.W, self.b
    #     # Weights are in columns as w1 in column 1, w2 in column 2 and so on.
    #     z = np.matmul(X, W) + b
    #     # print(z)
    #     f_wb = self._softmax(z)
    #     # print('fwb',f_wb)
    #     return f_wb

    def _one_hot_enc(self):
        _, c = self.W.shape
        y_hot = np.zeros((len(self.y), c))
        y_hot[np.arange(len(self.y)), self.y] = 1
        self.y_hot = y_hot
        return y_hot

    def _get_cost(self):
        X, W, b, y = self.X, self.W, self.b, self.y
        # Unfortunately below code won't work because of high possibility of log(0) errors.
        # t = time.time()
        # f_wb = self._get_f_wb()
        # y_hat = f_wb[np.arange(y.shape[0]), y]
        # j_wb = -np.log(y_hat)
        # cost = np.sum(j_wb)
        # print('t1', time.time() - t)

        # Below code works without log(0) errors
        # Weights are in columns as w1(28*28) in column 1, w2 in column 2 and so on.
        z = np.matmul(X, W) + b
        z_max = np.max(z, axis=1).reshape((z.shape[0], 1))
        z_new = z - z_max
        ez_new = np.exp(z_new)
        ez_sum = np.sum(ez_new, axis=1).reshape((-1, 1))
        log_softmax = z_new - np.log(ez_sum)
        loss = -log_softmax[np.arange(y.shape[0]), y]
        cost = np.mean(loss)
        return cost

    def _get_gradient(self):
        X, y, W, b, y_hot = self.X, self.y, self.W, self.b, self.y_hot
        z = np.matmul(X, W) + b
        f_wb = self._softmax(z)
        m, n = X.shape

        dj_dw = np.matmul(X.T, f_wb - y_hot) / m
        dj_db = np.sum(f_wb - y_hot, axis=0) / m

        return dj_dw, dj_db

    def _gradient_descent(self, alpha, epochs):
        # 'alpa' is learning rate
        for epoch in range(epochs):
            dj_dw, dj_db = self._get_gradient()
            self.W -= alpha * dj_dw
            self.b -= alpha * dj_db
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs or epoch == 0:
                print("Epoch {}, cost: {}".format(epoch+1, self._get_cost()))

    def optimize(self, alpha, epochs):
        self._gradient_descent(alpha=alpha, epochs=epochs)

    def predict(self, X):
        z = np.matmul(X, self.W) + self.b
        f_wb = self._softmax(z)
        return np.argmax(f_wb, axis=1)

    def normalize_data(self):
        X = self.X
        mu = np.mean(X)
        sigma = np.std(X)
        self.X = (X - mu) / sigma
        self.mu = mu
        self.sigma = sigma

    def get_normailizing_data(self):
        '''return -> mu, sigma'''
        return self.mu, self.sigma
