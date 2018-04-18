# -*- coding: utf-8 -*-
# @Author: qingshuizhiren
# @Date:   2018-03-30 18:16:11
# @Last Modified by:   qingshuizhiren
# @Last Modified time: 2018-04-01 11:47:49
# @E-mail: qingshuizhiren@foxmail.com
#

# import random
import numpy as np
# from sklearn import svm
from SVMofTwoClasses import svmoftwoclasses


def randsvm(dataplus, dataminus, rawnumofclf, colnumofclf, numofsample):
    clf = []
    index_plus = []
    index_minus = []
    # random to select the number of raw data for one
    for i in range(rawnumofclf):
        index_plus.append(np.random.randint(len(dataplus), size=numofsample))
    # random to select the number of raw data for zero
    for j in range(colnumofclf):
        index_minus.append(np.random.randint(len(dataminus), size=numofsample))
    # record all the SVM model
    for i in range(rawnumofclf):
        clf.append([])
        for j in range(colnumofclf):
            # 1000 data vs 1000 data , balance problem
            sample_plus = dataplus[index_plus[i], :]
            sample_minus = dataminus[index_minus[j], :]
            clf[i].append(svmoftwoclasses(sample_plus, sample_minus, False))
    return clf


def testrandsvm(clf, rawnumofclf, colnumofclf, test_data):
    # predict matrix
    predict = np.zeros((test_data.shape[0], rawnumofclf))
    for i in range(rawnumofclf):
        min_predict = np.ones((test_data.shape[0], rawnumofclf))
        for j in range(colnumofclf):
            # min_predict[:, j] = clf[i][j].predict_proba(test_data)[:, 0]
            min_predict[:, j] = clf[i][j].predict(test_data)
        predict[:, i] = np.min(min_predict, axis=1)
    ret_predict = np.max(predict, axis=1)
    return ret_predict

# def priorsvm():
