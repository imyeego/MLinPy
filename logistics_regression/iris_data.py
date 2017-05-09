# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
x = df.iloc[:, [0, 2]].values

plt.scatter(x[:50, 0], x[:50, 1],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(x[50:100, 0], x[50:100, 1],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(x[100:, 0], x[100:, 1],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc=2) # 把说明放在左上角，具体请参考官方文档
plt.show()

