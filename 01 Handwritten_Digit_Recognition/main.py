import putil
import numpy as np
import gzip, sys
import matplotlib.pyplot as plt


def get_image_data(path: str):
    with gzip.open(path, 'r') as file:
        magic_number = int.from_bytes(file.read(4), 'big')
        image_count = int.from_bytes(file.read(4), 'big')
        row_count = int.from_bytes(file.read(4), 'big')
        col_count = int.from_bytes(file.read(4), 'big')
        image_data = file.read(row_count * col_count * image_count)
        images = np.frombuffer(buffer=image_data, dtype=np.uint8).reshape((image_count, row_count, col_count))
        return images


def get_label_data(path: str):
    with gzip.open(path, 'r') as file:
        magic_number = int.from_bytes(file.read(4), 'big')
        label_count = int.from_bytes(file.read(4), 'big')
        labels_data = file.read()
        labels = np.frombuffer(buffer=labels_data, dtype=np.uint8)
        return labels


def get_train_data(num=None):
    images = get_image_data('./train-images-idx3-ubyte.gz')[:num]
    labels = get_label_data('./train-labels-idx1-ubyte.gz')[:num]
    return images, labels


def get_test_data():
    images = get_image_data('./t10k-images-idx3-ubyte.gz')
    labels = get_label_data('./t10k-labels-idx1-ubyte.gz')
    return images, labels


def main():
    x_train, y_train = get_train_data(2)
    outputs_n = 10
    # plt.imshow(x_train[2])
    # plt.show()
    # print(labels[2])
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    # print(x_train.shape)
    # print(y_train.shape)
    W = np.random.random((x_train.shape[1], outputs_n))
    b = np.random.random(outputs_n)
    # print(W.shape, b.shape)
    model = putil.Softmax()
    # model.fit(x_train, y_train, np.zeros((784, 10)), np.ones(10))
    model.fit(x_train, y_train, W, b)
    model._get_cost()
    model._get_gradient()
    # model.fit(np.arange(12).reshape((-1,3)), np.arange(4), np.zeros((3,4)), np.array([2,4,6,8]))
    # model._get_f_wb()
if __name__ == '__main__':
    main()
