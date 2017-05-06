#!/usr/bin/python
# _*_ coding:utf-8 _*_

import urllib.request

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

b_raw_data = urllib.request.urlopen(url).read() #返回字节字符串bytes类型
raw_data = str(b_raw_data,encoding = 'utf-8')
#raw_data = b_raw_data().decode("utf8")
#print(type(raw_data))

f = open("datasets.txt","w")
f.writelines(raw_data)
f.close()
