#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 00:58:46 2019

@author: sl
"""
from smlib.logistic_regression.logreg import LogisticRegression
from sklearn.linear_model import LogisticRegression as skLR

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=20000,
                           n_features=20,
                           n_informative=10,
                           n_classes=2)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

lr = LogisticRegression(solver='newton-cg', fit_intercept=True, C=1, n_iters=300)
lr.fit(X_train, y_train)
print('\nsmlib LogReg')
print(classification_report(y_train, lr.predict(X_train)))
print(classification_report(y_test, lr.predict(X_test)))


sklr = skLR(solver='newton-cg', fit_intercept=True)
sklr.fit(X_train, y_train)
print('\nsklearn LogReg')
print(classification_report(y_train, sklr.predict(X_train)))
print(classification_report(y_test, sklr.predict(X_test)))
