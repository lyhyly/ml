import mnist
import mnist_standardized as standardized
import numpy as np
import time
import json

#反向传播

#计算S函数导数
def sigmoid_gradient(sigmoid):

    return np.multiply(sigmoid,1-sigmoid)

"""
反向传播梯度计算
"""
def back(X, Y, y_hat, w2, h):
    w2_gradient = np.matmul(prepend_bias(h).T,(y_hat-Y)) / X.shape[0]
    #L关于h的导数
    L_x_w1 = np.matmul(y_hat - Y, w2[1:].T)
    #sig(x*w1)的导数
    h_gradient = sigmoid_gradient(h)
    #根据链式求导法则，计算L关于w1的导数
    #L关于h的导数 乘以 h关于sig(x*w1)的导数 乘以 x*w1关于w1的导数
    w1_gradient = np.matmul(prepend_bias(X).T, L_x_w1 * h_gradient ) / X.shape[0]
    return (w1_gradient,w2_gradient)

"""
初始化权重矩阵
n_input_variables -> 输入样本数据列数
n_hidden_nodes -> 中间层列数
n_classes -> w2列数
"""
def initialize_weights(n_input_variables, n_hidden_nodes, n_classes):
    #样本列+偏置列
    w1_rows = n_input_variables + 1
    #权重初始化公式
    #random.randn 生成标准正态分布矩阵
    #再根据《+- 1/r开平方》的权重初始化公式进行缩放
    w1 = np.random.randn(w1_rows,n_hidden_nodes) * np.sqrt(1 / w1_rows)

    #隐藏节点列+偏置列
    w2_rows = n_hidden_nodes + 1
    w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)
    return (w1, w2)

"""
训练函数
X_train ->训练样本
Y_train ->训练样本结果
X_test ->测试样本
Y_train ->测试样本结果
n_hidden_nodes ->中间节点数量
iterations ->学习次数
lr ->学习率

"""
def train1(X_train, Y_train, X_test, Y_test, n_hidden_nodes, epochs, batch_size, lr):
    #训练样本列数
    n_input_variables = X_train.shape[1]
    #结果列数 = w2列数
    n_classes = Y_train.shape[1]
    #初始化权重矩阵
    w1,w2 = initialize_weights(n_input_variables,n_hidden_nodes,n_classes)
    #样本分批
    x_batches, y_batches = prepare_batches(X_train, Y_train, batch_size)
    accuracy = 0
    #迭代训练
    for epoch in range(epochs):
        for batch in range(len(x_batches)):
            #计算预测值以及中间值
            y_hat, h = forward(x_batches[batch], w1, w2)
            #计算梯度
            w1_gradient, w2_gradient = back(x_batches[batch], y_batches[batch], y_hat, w2, h)
            #梯度下降
            w1 = w1 - (w1_gradient * lr)
            w2 = w2 - (w2_gradient * lr)
            #计算训练后的数据，并打印损失
            accuracy = report(epoch, batch, X_train, Y_train, X_test, Y_test, w1, w2)



    return (w1, w2,accuracy)

"""
样本分批
"""
def prepare_batches(X_train, Y_train, batch_size):
    x_batches = []
    y_batches = []
    #训练集数量
    n_examples = X_train.shape[0]
    #起始值，总长，步长
    for batch in range(0, n_examples, batch_size):
        batch_end = batch+batch_size
        x_batches.append(X_train[batch:batch_end])
        y_batches.append(Y_train[batch:batch_end])
    return (x_batches,y_batches)
#正向传播

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
    return (y_hat,h)

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
    y_hat,h = forward(X,w1,w2)
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
def report(epoch, batch, X_train, Y_train, X_test, Y_test, w1, w2):
    #预测函数
    y_hat,h = forward(X_train,w1,w2)
    #计算损失
    training_loss = loss(Y_train , y_hat)
    #使用训练结果计算测试数据
    classifications = classify(X_test,w1,w2);
    #计算准确率
    #86.3
    accuracy = np.average(classifications == Y_test) * 100.0
    #打印结果
    print("%5d-%d > Loss: %.6f,Accuracy: %.2f%%" %
          (epoch, batch, training_loss, accuracy))
    return accuracy

if __name__ == "__main__":
    st = time.time()
    w1, w2, accuracy1 = train1(mnist.X_train, mnist.Y_train, mnist.X_test, mnist.Y_test, n_hidden_nodes=200,
                               iterations=10000, lr=0.01)
    end = time.time()
    print("train1耗时：%.5f,准确率：Accuracy: %.2f%%" % (end - st, accuracy1))
