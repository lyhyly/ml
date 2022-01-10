# An MNIST loader.

import numpy as np
import gzip
import struct
"""
加载后进行标准化
"""
#将压缩包解压，并将图片转为矩阵
def load_images(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Read the header information into a bunch of variables
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all the pixels into a NumPy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)

def standardize(training_set, test_set):
    average = np.average(training_set)
    standardize_deviation = np.std(training_set)
    training_set_standardized = (training_set - average) / standardize_deviation
    test_set_standardized = (test_set - average) / standardize_deviation
    return (training_set_standardized,test_set_standardized)

# 60000 images, each 785 elements (1 bias + 28 * 28 pixels)
X_train_raw = load_images("../6/train-images-idx3-ubyte.gz")

# 10000 images, each 785 elements, with the same structure as X_train
X_test_raw = load_images("../6/t10k-images-idx3-ubyte.gz")

X_train, X_test_all = standardize(X_train_raw,X_test_raw)
X_validation, X_test = np.split(X_test_all,2)


def load_labels(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all the labels into a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)



def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y


# 60K labels, each with value 1 if the digit is a five, and 0 otherwise
Y_train = one_hot_encode(load_labels("../6/train-labels-idx1-ubyte.gz"))

# 10000 labels, with the same encoding as Y_train
Y_test_raw = load_labels("../6/t10k-labels-idx1-ubyte.gz")
Y_validation, Y_test = np.split(Y_test_raw, 2)
