# -*- coding: utf-8 -*-
# @Author: qingshuizhiren
# @Date:   2018-03-28 15:51:15
# @Last Modified by:   qingshuizhiren
# @Last Modified time: 2018-03-31 10:23:46
# @E-mail: qingshuizhiren@foxmail.com

# from sklearn import datasets
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# print(digits.data)
# import random
import numpy as np
from sklearn import svm
# ran = np.random.randint(0,10,[10,10])
# print(ran)
# index = np.where(ran == 1)
# print(index)
# print(index[0], index[1])
# print(len(index),len(index[0]))

# =============================================================================
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC(probability=True)
# clf.fit(X, y)
# p1 = clf.predict_proba([[2., 2.], [1, -1], [3, 4]])
# p2 = clf.predict_proba([[-2., 2.], [-1, -1], [3, -4]])
# print(p1, '\n')
# print(p2, '\n')
# q1 = p1[:, 0].reshape(-1, 1)
# q2 = p2[:, 0].reshape(-1, 1)
# q = np.concatenate((q1, q2), axis=1)
# print(q1, '\n')
# print(q2, '\n')
# print(q)
# 
# =============================================================================
# randmat1 = np.random.randint(10,size=[4,5])
# randmat2 = np.random.randint(10,size=[4,5])
# print(randmat1)
# print(randmat2)
# minrandmat = min(randmat1, randmat2)
# print(minrandmat)
# print(randmat1[0])
test_data = np.load('data/test_data.npy')
test_label = np.load('data/test_label.npy')
train_data = np.load('data/train_data.npy')
train_label = np.load('data/train_label.npy')
# print(test_label)