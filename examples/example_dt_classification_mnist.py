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
from sklearn.tree import DecisionTreeClassifier

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_leaf=5)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# now fit our own decision tree and compare quality with sklearn
dt = DecisionTree(criterion='gini', max_depth=7, min_samples_leaf=5)
dfX = pd.DataFrame(data)
dfy = pd.Series(digits.target)

print('fitting')
dt.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
predicted = dt.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (dt, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
