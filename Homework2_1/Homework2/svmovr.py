# -*- coding: utf-8 -*-
# @Author: qingshuizhiren
# @Date:   2018-03-30 15:43:43
# @Last Modified by:   qingshuizhiren
# @Last Modified time: 2018-04-01 10:44:17
# @E-mail: qingshuizhiren@foxmail.com
#

import numpy as np
from sklearn import svm, preprocessing
from sklearn.model_selection import GridSearchCV

test_data = np.load('data/test_data.npy')
test_label = np.load('data/test_label.npy')
train_data = np.load('data/train_data.npy')
train_label = np.load('data/train_label.npy')
scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# parameters = {'kernel': ('rbf', 'linear'), 'C': [1, 5, 10]}
# svr = svm.SVC()
# clf = GridSearchCV(svr, parameters)
# clf.fit(train_data, train_label)
# print(clf.best_estimator_)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(train_data, train_label)
clf.predict(test_data)
acc = clf.score(test_data, test_label)
print(acc)
