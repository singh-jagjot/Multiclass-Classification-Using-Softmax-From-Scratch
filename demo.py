import numpy as np
import cv2
import matplotlib.pyplot as plt
import Softmax, pickle, os

def load_data(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data
# a = np.arange(9).reshape((3,3))
# b = np.max(a, axis=1).reshape((a.shape[0], 1))
# print(a)
# # print(b)
# # print(a - b)
# c = np.sum(a, axis=1).reshape((-1, 1))
# print(c)
# print(a/c)
# a = np.arange(5)
# b = np.array([0,11,22,3,4])
# print(a==b)

# with open('one.jpg', 'rb') as img1:
#     file = img1.read()
#     image = np.frombuffer(buffer=file, dtype=np.uint8).reshape((28,28,3))  
# plt.imshow(image, format='jpg')
# plt.show()
# print(images.shape)
img = cv2.imread('one.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
img = img < 80
plt.imshow(img, cmap='gray')
plt.show()
img = img.reshape((1, 28 * 28))
FILE_NAME = 'data0.pkl'
W, b = load_data(FILE_NAME)
model = Softmax.Softmax()
model.set_weight(W)
model.set_bias(b)
# img = img / 255
test_predictions = model.predict(img)
# test_accuracy = get_accuracy(y_test, test_predictions)
# print('Accuracy: ', test_accuracy, "%")
print(test_predictions)

# print(images.shape)