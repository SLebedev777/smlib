# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:51:10 2019

@author: NRA-LebedevSM
"""

# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from smlib.knn import kNN
from smlib.decision_trees.dt import DecisionTree
from smlib.model_evaluation.bias_variance import bias_variance_regression
from smlib.model_evaluation.metrics import mse

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
# Add noise to targets
y[::5] += .001 * (0.5 - np.random.rand(8))

T = np.linspace(0, 5, 500)[:, np.newaxis]
yT = np.sin(T).ravel()
yT[::5] += .001 * (0.5 - np.random.rand(100))
'''
T  = np.array([X[10]])
yT = np.array([y[10]])
'''
###############################################################################
plt.figure(figsize=(10,5))
plt.scatter(X, y, c='k', label='data')

models = [
        (neighbors.KNeighborsRegressor(n_neighbors=5), 'g', 'sklearn kNN 5'),
        (kNN(task='regression', k=30, metric='l2'), 'y', 'smlib kNN 30'), 
        (kNN(task='regression', k=10, metric='l2'), 'r', 'smlib kNN 10'), 
        (kNN(task='regression', k=1, metric='l2'), 'b', 'smlib kNN 1')
        ]

for knn, color, label in models:
    y_ = knn.fit(X, y).predict(T)
    plt.plot(T, y_, c=color, label=label)
plt.legend()
plt.title('KNeighbors regression example')

plt.show()


# bias-variance analysis w.r.t. changing model complexity (number of neighbors)
complexity_param = range(1, 10)
models = [kNN(task='regression', k=k, metric='l2') for k in complexity_param]
   
EPE, B, V = bias_variance_regression(models, X, y, T, yT, n_subsamples=30)
    
plt.figure(figsize=(10, 5))
plt.plot(complexity_param, EPE, c='r', label='avg(EPE)')
plt.plot(complexity_param, B,   c='b', label='avg(B**2)')
plt.plot(complexity_param, V,   c='g', label='avg(V)')

plt.legend()
plt.show()

###################################################
#comparison with decision trees
complexity_param = range(1, 10)
models = [DecisionTree(task='regression',
                       criterion='mse',
                       max_depth=k) for k in complexity_param]
   
EPE, B, V = bias_variance_regression(models, X, y, T, yT, n_subsamples=30)
    
plt.figure(figsize=(10, 5))
plt.plot(complexity_param, EPE, c='r', label='avg(EPE)')
plt.plot(complexity_param, B,   c='b', label='avg(B**2)')
plt.plot(complexity_param, V,   c='g', label='avg(V)')

plt.legend()
plt.show()
