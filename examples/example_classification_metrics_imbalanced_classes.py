# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 00:55:51 2019

@author: Семен
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from smlib.model_evaluation.classification_metrics import (
        BinaryClassificationMetrics, roc_curve, precision_recall_curve)
import pandas as pd

def plot_hyperplane(clf, ax, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x, max_x)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    ax.plot(xx, yy, linestyle, label=label)


def classification_case(X, y, clf, xlim, ylim, title):
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(221)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.3,
                                                    random_state=105)
    clf.fit(X_train, y_train)

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, marker='o', 
               alpha=0.5, label='train points')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=60, marker='X',
               label='test points')
    ax.legend()
    ax.title.set_text(title)
    for est in clf.estimators_:
        plot_hyperplane(est, ax, xlim[0], xlim[1], 'k--', '')
    
    last_label = 2
    y_test_pred = clf.estimators_[2].predict(X_test)*last_label  # emulate 0-2 binary answer
    y_test_probas = clf.estimators_[2].predict_proba(X_test)[:, 1]
    m = BinaryClassificationMetrics(y_test, y_test_pred, pos_label=last_label)
 
    print('='*50 + '\n')
    print(title)
    print(f'Estimator: {clf}')
    print('='*50)
    print(pd.Series(y_test).value_counts())
    print('\nMetrics of binary classification class2 (yellow) vs rest:')    
    print('\nconfusion_matrix:')
    print(m.confusion_matrix)
    print(f'accuracy: {m.accuracy}')
    print(f'precision: {m.precision}')
    print(f'recall: {m.recall}')
    print(f'f1: {m.f1}')
    print(f'balanced_accuracy: {m.balanced_accuracy}')

    fpr, tpr, t = roc_curve(y_test, y_test_probas, pos_label=last_label)
    thres05_index = np.argmin(np.abs(np.array(t) - 0.5))
    ax = plt.subplot(222)
    ax.title.set_text('ROC for classifying class2 vs rest')
    ax.plot(fpr, tpr)
    ax.axhline(tpr[thres05_index], 0, 1, color='r', linestyle='--', label='default threshold')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    ax.legend(loc='lower right')
    p, r, t, f = precision_recall_curve(y_test, y_test_probas, pos_label=last_label)
    thres05_index = np.argmin(np.abs(np.array(t) - 0.5))
    ax = plt.subplot(223)
    ax.title.set_text('PR and F1R curve for classifying class2 vs rest')
    ax.plot(r, p, label='Precision-Recall curve')
    ax.plot(r, f, color='g', label='F1-Recall curve')
    ax.axvline(r[thres05_index], 0, 1, color='r', linestyle='--', label='default threshold')
    ax.axhline(p[thres05_index], 0, 1, color='r', linestyle='--')
    best_f1_index = np.argmax(f)
    ax.axvline(r[best_f1_index], 0, 1, color='g', linestyle='--', label='optimal threshold')
    plt.xlabel('recall')
    plt.ylabel('precision (f1)')
    ax.legend()
    plt.show()
    
M = 3*1000
X, y = make_classification(n_samples=M,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=3,
                           n_clusters_per_class=1,
                           shuffle=False,
                           random_state=105)
min_x = np.min(X[:, 0])
max_x = np.max(X[:, 0])

min_y = np.min(X[:, 1])
max_y = np.max(X[:, 1])

models = [LogisticRegression(solver='newton-cg'),
          SVC(kernel='linear', probability=True)
        ]

for model in models:
    clf_balanced = OneVsRestClassifier(model)
    classification_case(X, y, clf_balanced,
                        [min_x, max_x], [min_y, max_y], 'Balanced classes')
    
    
    ###############################################################################
    Ximb = X[:int(.7*M)]  # make 3rd (yellow) class highly imbalanced (rare)
    yimb = y[:int(.7*M)]
    
    
    clf_imbalanced = OneVsRestClassifier(model)
    classification_case(Ximb, yimb, clf_imbalanced,
                        [min_x, max_x], [min_y, max_y], 
                        'Imbalanced classes, no class weight')
    
    ###############################################################################
    model.set_params(**{'class_weight': 'balanced'})
    clf_imb2 = OneVsRestClassifier(model)
    classification_case(Ximb, yimb, clf_imb2,
                        [min_x, max_x], [min_y, max_y], 
                        'Imbalanced classes, with class weight')
