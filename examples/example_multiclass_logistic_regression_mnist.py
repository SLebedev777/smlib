# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:48:19 2019

@author: NRA-LebedevSM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smlib.logistic_regression.logreg import MulticlassLogisticRegression

from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression as skLR

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = skLR(multi_class='multinomial', solver='newton-cg')

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# now fit our own decision tree and compare quality with sklearn
lr = MulticlassLogisticRegression(alpha=0.01)

print('fitting')
lr.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
predicted = lr.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (lr, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
