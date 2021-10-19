import mnist as data
import numpy as np
import time;
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
st = time.time()
w = train(data.X_train,data.Y_train,iterations=100,lr=1e-5)
test(data.X_test,data.Y_test,w)
end = time.time()
print("耗时：%.5f" % (end-st))