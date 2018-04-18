# -*- coding: utf-8 -*-
# @Author: qingshuizhiren
# @Date:   2018-03-30 18:16:11
# @Last Modified by:   qingshuizhiren
# @Last Modified time: 2018-03-31 10:56:47
# @E-mail: qingshuizhiren@foxmail.com
#

# import random
import numpy as np
from sklearn import svm
from SVMofTwoClasses import svmoftwoclasses

def randsvm(dataplus, dataminus, rawnumofclf, colnumofclf, numofsample):
    clf = []
    index_plus = []
    index_minus = []
    for i in range(rawnumofclf):
        index_plus.append(np.random.randint(len(dataplus), size=numofsample))
    for j in range(colnumofclf):
        index_minus.append(np.random.randint(len(dataminus), size=numofsample))
    for i in range(rawnumofclf):
        clf.append([])
        for j in range(colnumofclf):
            sample_plus = dataplus[index_plus[i], :]
            sample_minus = dataminus[index_minus[j], :]
            clf[i].append(svmoftwoclasses(sample_plus, sample_minus, True))
    return clf

def testrandsvm(clf, rawnumofclf, colnumofclf, test_data):
    predict = np.zeros((test_data.shape[0], rawnumofclf))
    for i in range(rawnumofclf):
        min_predict = np.ones((test_data.shape[0], rawnumofclf))
        for j in range(colnumofclf):
            min_predict[:, j] = clf[i][j].predict_proba(test_data)[:, 0]
        predict[:, i] = np.min(min_predict, axis=1)
    ret_predict = np.max(predict, axis=1)
    return ret_predict
