import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

X,Y = np.loadtxt('../../code/02_first/pizza.txt',skiprows=1,unpack=True)

plt.axis([0,50,0,50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel("预定数量")
plt.ylabel("披萨销售数量")

plt.plot(X, Y, "bo")

plt.show()