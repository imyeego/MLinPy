#!/usr/bin/python
# _*_ coding:utf-8 _*_

import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

#数据导入
data = []
labels = []

with open("datasets.txt","r") as file:
	for line in file:
		slices = line.strip().split(' ')
		data.append([float(sl) for sl in slices[:-1]]) #注意将字符型数据转化成浮点型
		labels.append(slices[-1])

x = np.array(data)
y_labels = np.array(labels)
y = np.zeros(y_labels.shape)


#标签转换为0/1
y[y_labels == 'fat'] = 1

#拆分训练数据与测试数据

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#使用信息熵作为特征选择标准
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
print(clf)
clf.fit(x_train, y_train)

#将树的结构写入文件
with open("tree.dot","w") as f:
	f = tree.export_graphviz(clf, out_file = f)


#打印特征系数
print(clf.feature_importances_)

#测试结果的打印
answer = clf.predict(x_train)
print(x_train)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

#准确率与召回率
print(clf.predict_proba(x))
answer_proba = clf.predict_proba(x)[:,-1] #取矩阵倒数第一列
print(classification_report(y, answer_proba,target_names = ['thin','fat']))











