# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:57:37 2020

@author: Семен
"""

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.model_selection import train_test_split

from smlib.bayes.lda import LinearDiscriminant as LDA
from smlib.model_evaluation.classification_metrics import BinaryClassificationMetrics
# #############################################################################
# Colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


# #############################################################################
# Generate datasets
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

X, y = dataset_fixed_cov()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X0 = X_train[y_train==0]
X1 = X_train[y_train==1]

plt.scatter(X0[:, 0], X0[:, 1], marker='.', color='red')
plt.scatter(X1[:, 0], X1[:, 1], marker='.', color='blue')

# class 0 and 1 : areas
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                     np.linspace(y_min, y_max, ny))


sklda = skLDA()
lda = LDA()

for clf, color, name in [(sklda, 'g', 'sklearn LDA'),
                         (lda, 'r', 'smlib LDA')
                         ]:
    clf.fit(X_train, y_train)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors=color)

    y_pred = clf.predict(X_test)
    m = BinaryClassificationMetrics(y_test, y_pred)
    print(f'Confusion matrix for {name}:')
    print(m.confusion_matrix)
    
        