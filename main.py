import Softmax
import numpy as np
import pickle
import gzip


def get_image_data(path: str):
    with gzip.open(path, 'r') as file:
        magic_number = int.from_bytes(file.read(4), 'big')
        image_count = int.from_bytes(file.read(4), 'big')
        row_count = int.from_bytes(file.read(4), 'big')
        col_count = int.from_bytes(file.read(4), 'big')
        image_data = file.read(row_count * col_count * image_count)
        images = np.frombuffer(buffer=image_data, dtype=np.uint8).reshape(
            (image_count, row_count, col_count))
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


def get_accuracy(y: np.ndarray, y_hat: np.ndarray):
    return np.sum(y == y_hat) / len(y) * 100


def save_data(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)


def load_data(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


def main():
    # Getting training data
    # (x_train : 60000 images, shape = (60000, 28, 28))
    # (y_train : 60000 image labels, shape = (60000,))
    x_train, y_train = get_train_data()

    # Number of classes (0,1,2 ... 9)
    c = 10

    # Setting training data(x_train) shape to (60000, 28*28)
    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

    # Normalize x_train
    x_train = x_train / 255.0

    # Setting weights and bias with random values
    m, n = x_train.shape
    # np.random.seed(10)
    W = np.random.random((n, c))
    # np.random.seed(10)
    b = np.random.random(c)

    # Initializing ML model
    model = Softmax.Softmax()

    # Setting up model with required parameters
    # labels data(y_train) will be one-hot encoded
    model.fit(x_train, y_train, W, b)

    # Starting 'Batch Gradient Descent'
    model.optimize(alpha=1, epochs=750)

    # Saving weights and bias for later use if needed.
    # data = model.get_weights_bias()
    # FILE_NAME = 'data0.pkl'
    # save_data(data, FILE_NAME)

    # Getting test data
    # (x_test : 10000 images, shape = (10000, 28, 28))
    # (y_test : 10000 image labels, shape = (10000,))
    x_test, y_test = get_test_data()

    # Setting testing data(x_test) shape to (10000, 28*28)
    x_test = np.reshape(
        x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    # Normalize x_test
    x_test = x_test / 255.0

    # Getting predictions on both test and training data
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    # Getting accuracy of the predictions
    train_accuracy = get_accuracy(y_train, train_predictions)
    test_accuracy = get_accuracy(y_test, test_predictions)

    # Comparing accuracy of training data with testing data
    print("Train Accuracy: {:.3f}%\nTest Accuracy: {:.3f}%".format(
        train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()
