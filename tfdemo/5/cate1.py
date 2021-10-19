import numpy as np


def predict(X, w):
    return np.matmul(X, w)


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

#梯度函数
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

#将预测值压缩
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#预测函数
def forward(X, w):
    return sigmoid(np.matmul(X, w))

#取整
def classify(X, w):
    return np.round(forward(X, w))
#问题损失函数
def mes_loss(X, Y, w):
    return np.average((forward(X, w) - Y) ** 2)

#新损失函数
def new_loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)




def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, new_loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" %
          (correct_results, total_examples, success_percent))

# Prepare data
x1, x2, x3, y = np.loadtxt("police.txt", skiprows=1, unpack=True)
#X = np.column_stack((np.ones(x1.size), x1, x2, x3))
X = np.column_stack((np.ones(x1.size), x1, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)

# Test it
test(X, Y, w)

# Print weights:
print(w.T)