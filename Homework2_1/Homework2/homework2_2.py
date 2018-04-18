# -*- coding: utf-8 -*-
# @Author: qingshuizhiren
# @Date:   2018-03-30 16:28:26
# @Last Modified by:   qingshuizhiren
# @Last Modified time: 2018-04-01 11:47:38
# @E-mail: qingshuizhiren@foxmail.com
#

import os
import numpy as np
from RandM3SVM import randsvm, testrandsvm
from sklearn import preprocessing
from sklearn.externals import joblib

# Read data
test_data = np.load('data/test_data.npy')
test_label = np.load('data/test_label.npy')
train_data = np.load('data/train_data.npy')
train_label = np.load('data/train_label.npy')

scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Find indices for -1, 1 and 0 separately
index_minus1 = np.where(train_label == -1)[0]
index_zero = np.where(train_label == 0)[0]
index_plus1 = np.where(train_label == 1)[0]

print(len(index_minus1), len(index_zero), len(index_plus1))
print(index_minus1.shape)
train_data_minus1 = np.array(train_data[index_minus1])
print(train_data_minus1.shape)

# Split train_data and train_label for different kinds
train_data_minus1 = np.array(train_data[index_minus1])
train_data_zero = np.array(train_data[index_zero])
train_data_plus1 = np.array(train_data[index_plus1])

train_data_non_minus1 = np.concatenate((train_data_plus1, train_data_zero))
train_data_non_zero = np.concatenate((train_data_minus1, train_data_plus1))
train_data_non_plus1 = np.concatenate((train_data_minus1, train_data_zero))

rawnumofclf = 5
colnumofclf = 2
numofsample = 1000
print('rawnumofclf = %d, colnumofclf = %d, numofsample = %d' % (rawnumofclf, colnumofclf, numofsample))

if os.path.exists("model/2_2_random/clf_m1.pkl"):
    clf_m1 = joblib.load('model/2_2_random/clf_m1.pkl')
else:
    print(1)
    clf_m1 = randsvm(train_data_minus1, train_data_non_minus1, rawnumofclf, colnumofclf, numofsample)
if os.path.exists("model/2_2_random/clf_z.pkl"):
    clf_z = joblib.load('model/2_2_random/clf_z.pkl')
else:
    print(2)
    clf_z = randsvm(train_data_zero, train_data_non_zero, rawnumofclf, colnumofclf, numofsample)
if os.path.exists("model/2_2_random/clf_p1.pkl"):
    clf_p1 = joblib.load('model/2_2_random/clf_p1.pkl')
else:
    print(3)
    clf_p1 = randsvm(train_data_plus1, train_data_non_plus1, rawnumofclf, colnumofclf, numofsample)

# clf_m1 = randsvm(train_data_minus1, train_data_non_minus1, rawnumofclf, colnumofclf, numofsample)
# clf_z = randsvm(train_data_zero, train_data_non_zero, rawnumofclf, colnumofclf, numofsample)
# clf_p1 = randsvm(train_data_plus1, train_data_non_plus1, rawnumofclf, colnumofclf, numofsample)

predict_m1 = testrandsvm(clf_m1, rawnumofclf, colnumofclf, test_data).reshape(-1, 1)
predict_z = testrandsvm(clf_z, rawnumofclf, colnumofclf, test_data).reshape(-1, 1)
predict_p1 = testrandsvm(clf_p1, rawnumofclf, colnumofclf, test_data).reshape(-1, 1)

predict = np.concatenate((predict_m1, predict_z, predict_p1), axis=1)
predict_label = np.argmax(predict, axis=1) - 1
correct_prediction = np.equal(test_label, predict_label)
accuracy = np.mean(correct_prediction, dtype=np.float32)
print(accuracy)
# acc = clf.score(test_data, test_label)
