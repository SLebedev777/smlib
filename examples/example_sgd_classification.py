# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:11:31 2020

@author: Семен
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from smlib.sgd.sgd import SGDClassifier
import numpy as np

# we create 50 separable points
X, Y = make_blobs(n_samples=100, centers=2, random_state=17, cluster_std=0.60)

# fit the model
clf = SGDClassifier(loss="hinge", alpha=.1, max_iter=5000, eta0=0.00001, tol=0.0001)

clf.fit(X, Y)
print(clf.coef_)
print(clf.intercept_)

# plot the line, the points, and the nearest vectors to the plane
xx = np.linspace(-8, 8, 10)
yy = np.linspace(-8, 8, 10)

X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = clf.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ['dashed', 'solid', 'dashed']
colors = 'k'
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
            edgecolor='black', s=20)

plt.axis('tight')
plt.show()