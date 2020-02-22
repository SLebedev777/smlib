# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:52:35 2020

@author: Семен
"""
import numpy as np
from sklearn.datasets import make_moons, make_classification
import matplotlib.pyplot as plt

from smlib.boosting.gb_class import GBClassifier

X, Y = make_moons(n_samples=100, noise=0.2, random_state=10)
#X, Y = make_classification(n_samples=100, n_features=2, n_redundant=0,
#                    n_informative=2, #random_state=2,
#                    n_clusters_per_class=1)

Y[Y == 0] = -1

# fit the model
params = {'loss': 'log', 'n_estimators': 100, 'max_depth': 1, 'learning_rate': 0.5}
clf = GBClassifier(**params)
clf.fit(X, Y)

h = .2
# create the grid for background colors
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, alpha=.8)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

levels = [0.0,]
linestyles = ['solid']
colors = 'k'
plt.contour(xx, yy, Z, levels, colors=colors, linestyles=linestyles)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
            edgecolor='black', s=20)

plt.axis('tight')
plt.show()