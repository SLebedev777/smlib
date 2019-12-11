#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 00:58:46 2019

@author: sl
"""
from smlib.logistic_regression.logreg import LogisticRegression
from smlib.model_evaluation.classification_metrics import (
        BinaryClassificationMetrics, roc_auc_score)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics as skmetrics

X, y = make_classification(n_samples=20000,
                           n_features=20,
                           n_informative=10,
                           n_classes=2)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

lr = LogisticRegression(solver='newton-cg', fit_intercept=True, C=1, n_iters=300)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

m = BinaryClassificationMetrics(y_test, y_pred)

print('\nconfusion_matrix')
print(skmetrics.confusion_matrix(y_test, y_pred))
print(m.confusion_matrix)

print('\naccuracy:')
print(skmetrics.accuracy_score(y_test, y_pred))
print(m.accuracy)

print('\nprecision')
print(skmetrics.precision_score(y_test, y_pred))
print(m.precision)

print('\nrecall')
print(skmetrics.recall_score(y_test, y_pred))
print(m.recall)

print('\nf1')
print(skmetrics.f1_score(y_test, y_pred))
print(m.f1)

print('\nbalanced_accuracy')
print(skmetrics.balanced_accuracy_score(y_test, y_pred))
print(m.balanced_accuracy)

y_pred_probas = lr.predict_proba(X_test)

print('\nROC AUC')
print(skmetrics.roc_auc_score(y_test, y_pred_probas))
print(roc_auc_score(y_test, y_pred_probas))

