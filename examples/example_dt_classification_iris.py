# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:48:19 2019

@author: NRA-LebedevSM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smlib.decision_trees.dt import DecisionTree

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

iris = load_iris()


plt.figure(1, (15, 10))
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]
                                ]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    print('-'*50)
    print('feature importances for: ')
    print(iris.feature_names[pair[0]], iris.feature_names[pair[1]])
       
    dt = DecisionTree(criterion='gini', max_depth=5, min_samples_leaf=2)
    dt.fit(X, y)
    print(dt.feature_importances_)
    
    skdt = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=2)
    skdt.fit(X, y)
    print(skdt.feature_importances_)
    
    plt.subplot(2, 3, pairidx + 1)
    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
