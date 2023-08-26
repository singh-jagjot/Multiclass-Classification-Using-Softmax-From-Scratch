import putil
import numpy as np
import pickle
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

def get_accuracy(y:np.ndarray, y_hat:np.ndarray):
    return np.sum(y == y_hat)/ len(y) * 100

def save_data(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def load_data(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data

def main():
    x_train, y_train = get_train_data()
    # Number of classes
    c = 10
    # plt.imshow(x_train[2])
    # plt.show()
    # print(labels[2])
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_train = x_train / 255.0
    m, n = x_train.shape
    np.random.seed(10)
    W = np.random.random((n, c))
    np.random.seed(10)
    b = np.random.random(c)

    model = putil.Softmax()
    model.fit(x_train, y_train, W, b)
    # model.normalize_data()
    model.optimize(1, 2000)

    data = model.get_weights_bias()
    FILE_NAME = 'data0.pkl'
    save_data(data, FILE_NAME)
    # W2, b2 = load_data(FILE_NAME)
    # model2 = putil.Softmax()
    # model2.set_weight(W2)
    # model2.set_bias(b2)
    # print(model._get_cost(), m._get_cost())
    # mu, sigma = model.get_normailizing_data()
    # x_train = (x_train - mu) / sigma
    x_test, y_test = get_test_data()
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    x_test = x_test / 255.0
    # x_test = (x_test - mu) / sigma

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    train_accuracy = get_accuracy(y_train, train_predictions)
    test_accuracy = get_accuracy(y_test, test_predictions)
    print("Train Accuracy: {:.3f}%\nTest Accuracy: {:.3f}%".format(train_accuracy, test_accuracy))

def main2():
    FILE_NAME = 'data0.pkl'
    W, b = load_data(FILE_NAME)
    model = putil.Softmax()
    model.set_weight(W)
    model.set_bias(b)

    x_test, y_test = get_test_data()
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    x_test = x_test / 255.0

    test_predictions = model.predict(x_test)
    test_accuracy = get_accuracy(y_test, test_predictions)
    print('Accuracy: ', test_accuracy, "%")
if __name__ == '__main__':
    # main()
    main2()
