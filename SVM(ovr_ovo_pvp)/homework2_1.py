# -*- coding: utf-8 -*-
# @Author: qingshuizhiren
# @Date:   2018-03-28 15:05:28
# @Last Modified by:   qingshuizhiren
# @Last Modified time: 2018-04-01 11:15:25
# @E-mail: qingshuizhiren@foxmail.com

# import random
# from sklearn import svm
import numpy as np
import os
from SVMofTwoClasses import svmoftwoclasses
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import manifold
import matplotlib.pyplot as plt

# Read data
test_data = np.load('E:/OneDrive/神经网络/homework/hm2/data_hw2/test_data.npy')
test_label = np.load('E:/OneDrive/神经网络/homework/hm2/data_hw2/test_label.npy')
train_data = np.load('E:/OneDrive/神经网络/homework/hm2/data_hw2/train_data.npy')
train_label = np.load('E:/OneDrive/神经网络/homework/hm2/data_hw2/train_label.npy')
# Standardize features by removing the mean and scaling to unit variance
scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# print(test_data.shape, test_label.shape, train_data.shape,
#       train_label.shape, test_data.dtype)
# clf = svm.SVC()
# clf.fit(train_data, train_label)
# print(clf.predict(test_data[0:100]))

# Find indices for -1, 1 and 0 separately
index_minus1 = np.where(train_label == -1)
index_plus1 = np.where(train_label == 1)
index_zero = np.where(train_label == 0)
# print(index_minus1[0], len(index_minus1[0]))
# print(index_plus1, len(index_plus1[0]))
# print(index_zero, len(index_zero[0]))

# Split train_data and train_label for different kinds
train_data_minus1 = np.array(train_data[index_minus1])
train_data_zero = np.array(train_data[index_zero])
train_data_plus1 = np.array(train_data[index_plus1])
# dimensionality reduction
train_data_minus1_embedded = manifold.TSNE().fit_transform(train_data_minus1)
train_data_zero_embedded = manifold.TSNE().fit_transform(train_data_zero)
train_data_plus1_embedded = manifold.TSNE().fit_transform(train_data_plus1)
# plot
plt.plot(train_data_minus1_embedded[:, 0], train_data_minus1_embedded[:, 1],'r.')
plt.plot(train_data_zero_embedded[:, 0], train_data_zero_embedded[:, 1],'go')
plt.plot(train_data_plus1_embedded[:, 0], train_data_plus1_embedded[:, 1],'b*')
plt.show()
# one vs rest
train_data_non_minus1 = np.concatenate((train_data_plus1, train_data_zero))
train_data_non_zero = np.concatenate((train_data_minus1, train_data_plus1))
train_data_non_plus1 = np.concatenate((train_data_minus1, train_data_zero))

# train_label_minus1 = np.array(train_label[index_minus1])
# train_label_plus1 = np.array(train_label[index_plus1])
# train_label_zero = np.array(train_label[index_zero])
# print(len(train_data_minus1), train_data_minus1)
# tra = np.concatenate((train_data_minus1, train_data_plus1))
# print(tra.shape)
# print(tra)

# Create and train 3 two-class SVMs for different kinds using one-versus-rest
# Print 1, 2, 3 and 4 to visualize the progress
# if os.path.exists("model/clf_m1.pkl"):
#     clf_m1 = joblib.load('model/clf_m1.pkl')
# else:
#     print(1)
#     clf_m1 = svmoftwoclasses(train_data_minus1, train_data_non_minus1, True)
# if os.path.exists("model/clf_z.pkl"):
#     clf_z = joblib.load('model/clf_z.pkl')
# else:
#     print(2)
#     clf_z = svmoftwoclasses(train_data_zero, train_data_non_zero, True)
# if os.path.exists("model/clf_p1.pkl"):
#     clf_p1 = joblib.load('clf_p1.pkl')
# else:
#     print(3)
#     clf_p1 = svmoftwoclasses(train_data_plus1, train_data_non_plus1, True)
#     print(4)
#
# predict = []
# predic_m1 = clf_m1.predict_proba(test_data)[:, 0].reshape(-1, 1)
# predic_z = clf_z.predict_proba(test_data)[:, 0].reshape(-1, 1)
# predic_p1 = clf_p1.predict_proba(test_data)[:, 0].reshape(-1, 1)
# predict = np.concatenate((predic_m1, predic_z, predic_p1), axis=1)
# predict_label = np.argmax(predict, axis=1) - 1
# correct_prediction = np.equal(test_label, predict_label)
# accuracy = np.mean(correct_prediction, dtype=np.float32)
# print(accuracy)

# load the trained SVM model if it exists, otherwise train a SVM
if os.path.exists("model/2_1/clf_m1.pkl"):
    clf_m1 = joblib.load('model/2_1/clf_m1.pkl')
else:
    print(1)
    # train svm and let the correct event be 1
    clf_m1 = svmoftwoclasses(train_data_minus1, train_data_non_minus1, False)
if os.path.exists("model/2_1/clf_z.pkl"):
    clf_z = joblib.load('model/2_1/clf_z.pkl')
else:
    print(2)
    clf_z = svmoftwoclasses(train_data_zero, train_data_non_zero, False)
if os.path.exists("model/2_1/clf_p1.pkl"):
    clf_p1 = joblib.load('model/2_1/clf_p1.pkl')
else:
    print(3)
    clf_p1 = svmoftwoclasses(train_data_plus1, train_data_non_plus1, False)
    print(4)

# Make predictions
predict = []
predic_m1 = clf_m1.predict(test_data).reshape(-1, 1)
predic_z = clf_z.predict(test_data).reshape(-1, 1)
predic_p1 = clf_p1.predict(test_data).reshape(-1, 1)
predict = np.concatenate((predic_m1, predic_z, predic_p1), axis=1)
predict_label = np.argmax(predict, axis=1) - 1
correct_prediction = np.equal(test_label, predict_label)
accuracy = np.mean(correct_prediction, dtype=np.float32)
print(accuracy)
