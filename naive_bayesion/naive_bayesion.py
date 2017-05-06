#!/usr/bin/python
# _*_ coding:utf-8 _*_

import time
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import tree
import numpy as np

#从txt文件中导入数据

datasets = np.loadtxt("datasets.txt", delimiter = ",")

x = datasets[:,:7]
y = datasets[:,8]

#使用高斯贝叶斯分类器
def gaussian_nb(a, b):
	print("\n--------------GaussianNB-------------")
	model = GaussianNB()
	start_time = time.time()
	model.fit(a, b)
	print('training took %fs!'%(time.time() - start_time))
	print(model)
	expected = b
	predicted = model.predict(a)
	print(metrics.classification_report(expected,predicted))
	print(metrics.confusion_matrix(expected,predicted))

def multinomial_nb(a, b):
	print("\n-------------MultinomialGB--------------")
	model = MultinomialNB(alpha = 1)
	start_time = time.time()
	model.fit(a, b)
	print('training took %fs!'%(time.time() - start_time))
	print(model)
	expected = b
	predicted = model.predict(a)
	print(metrics.classification_report(expected,predicted))
	print(metrics.confusion_matrix(expected,predicted))

def bernoulli_nb(a, b):
	print("\n-------------BernoulliGB--------------")
	model = BernoulliNB(alpha = 1,binarize = 0.0)
	start_time = time.time()
	model.fit(a, b)
	print('training took %fs!'%(time.time() - start_time))
	print(model)
	expected = b
	predicted = model.predict(a)
	print(metrics.classification_report(expected,predicted))
	print(metrics.confusion_matrix(expected,predicted))

gaussian_nb(x, y)
multinomial_nb(x, y)
bernoulli_nb(x, y)

'''
函数重构，解除代码冗余
def based_nb(a, b, f):
	print("\n--------------"+ f +"-------------")
	if f == 'GaussionNB':
		model = f()
	elif f == 'MultinomialNB':
		model = f(alpha = 1)
	elif f == 'BernoulliNB':
		model = f(alpha = 1,binarize = 0.0)
	else:
		exit()
 	start_time = time.time()
	model.fit(a, b)
	print('training took %fs!'%(time.time() - start_time))
	print(model)
	expected = b
	predicted = model.predict(a)
	print(metrics.classification_report(expected,predicted))
	print(metrics.confusion_matrix(expected,predicted))

based_nb(x, y, Gaussion)
based_nb(x, y, Multinomial)
based_nb(x, y, Bernoulli)

'''


