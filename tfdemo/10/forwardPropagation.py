import mnist as data
import numpy as np
import time
import json

#正向传播

#梯度函数
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

#激活函数
#将预测值压缩
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#机器学习：在线编写3层神经网络
#预测函数
"""
783对应一张图片的783个像素点(例：30*60)
X-> (6000,783) X 矩阵 -> (6000,200)

最终会得到(6000,10)；

"""
def forward(X, w1,w2):
    #初始层数据计算 需要与w1相乘
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat

#二层网络激活函数
def softmax(logits):
    #求指数
    exponentials = np.exp(logits)
    #根据公式计算最终值；在使用reshape转换格式
    return exponentials / np.sum(exponentials, axis = 1).reshape(-1,1)

#增加偏置列
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

#取整
def classify(X, w1,w2):
    y_hat = forward(X,w1,w2)
    labels = np.argmax(y_hat,axis=1)
    return labels.reshape(-1,1)


#新损失函数
def loss(Y_train, y_hat):
    return -np.sum(Y_train * np.log(y_hat)) / Y_train.shape[0]

"""
iteration 迭代次数
X_train 是训练数据
Y_train 是训练数据的打签结果   一张图片-》他的值是3 
X_test 测试数据源
Y_test 使用测试数据源计算后的结果
w1 -> S函数时使用的参数
w2 -> softmax函数参数
"""
def report(iteration,X_train,Y_train,X_test,Y_test,w1,w2):
    #预测函数
    y_hat = forward(X_train,w1,w2)
    #计算损失
    training_loss = loss(Y_train , y_hat)
    #使用训练结果计算测试数据
    classifications = classify(X_test,w1,w2);
    #计算准确率
    #86.3
    accuracy = np.average(classifications == Y_test) * 100.0
    #打印结果
    print("Iteration: %5d, Loss: %.6f,Accuracy: %.2f%%" %
          (iteration,training_loss,accuracy))

with open('weights.json') as f:
    weights = json.load(f)

w1,w2 = (np.array(weights[0]),np.array(weights[1]))

print("w1 height: %5d" % w1.shape[0])
print("X_train height: %5d" % data.X_train.shape[1])

report(0, data.X_train, data.Y_train, data.X_test, data.Y_test, w1, w2)
report(0, data.X_train, data.Y_train, data.X_test, data.Y_test, w1, w2)