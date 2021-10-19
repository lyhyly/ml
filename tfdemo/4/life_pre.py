import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict(X, w):
    return np.matmul(X, w)


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w


x1, x2, x3, y = np.loadtxt("life_expectancy.txt",skiprows=1,unpack=True)

X = np.column_stack((np.ones(x1.size), x1, x2, x3))

Y = y.reshape(-1, 1)
#步长过大会导致损失无法收敛
w = train(X, Y, iterations=1000000, lr=0.00001)

print("\n Weights: %s" % w.T)
print("\n A few predictions:")
for i in range(X.shape[0]):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))
