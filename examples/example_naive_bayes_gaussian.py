#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:10:53 2019

@author: sl
"""

from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB as skGNB
from smlib.bayes.nb import GaussianNB

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, y_train = data[:n_samples // 2], digits.target[:n_samples // 2]
X_test, y_test   = data[n_samples // 2:], digits.target[n_samples // 2:]


for classifier in [skGNB(), GaussianNB()]:
    classifier.fit(X_train, y_train)
   
    print(classifier)
    expected = y_test
    predicted = classifier.predict(X_test)
    print(classification_report(expected, predicted))
    
