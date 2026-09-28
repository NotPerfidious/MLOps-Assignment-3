from keras.datasets import fashion_mnist
import os
import numpy as np

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

os.makedirs("data/raw", exist_ok=True)

np.savez(
    "./data/raw/fashion_mnist.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)