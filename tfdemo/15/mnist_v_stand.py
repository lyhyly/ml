import mnist
import mnist_standardized as standardized
import neural_network as nn
import time
import json

st = time.time()
w1,w2,accuracy1 = nn.train1(mnist.X_train,mnist.Y_train, mnist.X_test, mnist.Y_test, n_hidden_nodes=200,epochs=2,batch_size=60, lr=0.01)
end = time.time()
print("mnist耗时：%.5f,准确率：Accuracy: %.2f%%" % (end-st, accuracy1))
st = time.time()
w1,w2,accuracy1 = nn.train1(standardized.X_train,standardized.Y_train, standardized.X_validation, standardized.Y_validation, n_hidden_nodes=200,epochs=2,batch_size=60, lr=0.01)
end = time.time()
print("stand耗时：%.5f,准确率：Accuracy: %.2f%%" % (end-st, accuracy1))
