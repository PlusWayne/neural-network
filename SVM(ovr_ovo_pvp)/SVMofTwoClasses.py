# -*- coding: utf-8 -*-
# @Author: qingshuizhiren
# @Date:   2018-03-28 16:42:50
# @Last Modified by:   qingshuizhiren
# @Last Modified time: 2018-03-28 19:30:04
# @E-mail: qingshuizhiren@foxmail.com
#

import numpy as np
from sklearn import svm


def svmoftwoclasses(dataplus, dataminus, prob):
    # label one to the data
    labelplus = np.ones(dataplus.shape[0])
    # lable zero to the data
    labelminus = np.zeros(dataminus.shape[0])
    # concatenate the label and data
    data = np.concatenate((dataplus, dataminus))
    label = np.concatenate((labelplus, labelminus))
    clf = svm.SVC(probability=prob)
    clf.fit(data, label)
    return clf
