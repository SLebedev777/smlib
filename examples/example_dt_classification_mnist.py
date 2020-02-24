# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:48:19 2019

@author: NRA-LebedevSM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smlib.decision_trees.dt import DecisionTree

from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier as sklearn_DecisionTree

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

dt_params = {'criterion': 'gini', 'max_depth': 7, 'min_samples_leaf': 5}
dt_classifiers = [
        sklearn_DecisionTree(**dt_params),
        DecisionTree(**dt_params)
        ]

for clf in dt_classifiers:
    print(f'fitting {clf}')
    
    # We learn the digits on the first half of the digits
    clf.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
    
    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = clf.predict(data[n_samples // 2:])
    
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
        
    print('8x8 color map of feature importances: ')
    plt.imshow(clf.feature_importances_.reshape((8, 8)))
    plt.show()