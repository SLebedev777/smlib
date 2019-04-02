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
from smlib.decision_trees import DecisionTree
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
train_errors = []
test_errors = []
model_complexity_parameters = range(50)  # list of parameter values, that change complexity
biases_conf_intervals = {'means': [], 'intervals': []}
variances_conf_intervals = {'means': [], 'intervals': []}
for k in model_complexity_parameters:
    model = kNN(task='regression', k=k, metric='l2')
    model.fit(X, y)
    y_ = model.predict(T)
    train_error = mse(y, model.predict(X))
    test_error = mse(yT, y_)
    train_errors.append(train_error)
    test_errors.append(test_error)
    bk, vk = bias_variance_regression(model, X_train=X, y_train=y, 
                                                 X_test=T, y_test=yT, 
                                                 n_subsamples=10, 
                                                 subsample_frac=.95)
    biases_conf_intervals['means'].append(bk.mean())
    biases_conf_intervals['intervals'].append(2*bk.std())
    variances_conf_intervals['means'].append(vk.mean())
    variances_conf_intervals['intervals'].append(2*vk.std())
    
plt.figure(figsize=(10, 7))
plt.plot(model_complexity_parameters, train_errors, c='g', label='train_errors')
plt.plot(model_complexity_parameters, test_errors, c='r', label='test_errors')
plt.legend()
plt.title('Train and test errors depending on model complexity')

plt.show()


plt.figure(figsize=(10, 7))
plt.errorbar(model_complexity_parameters, biases_conf_intervals['means'], 
             xerr=0.5,
             yerr=biases_conf_intervals['intervals'], linestyle='',
             c='g', label='bias confidence intervals')
plt.legend()
plt.title('Model bias^2 depending on model complexity')
plt.show()

plt.figure(figsize=(10, 7))
plt.errorbar(model_complexity_parameters, variances_conf_intervals['means'], 
             xerr=0.5,
             yerr=variances_conf_intervals['intervals'], linestyle='',
             c='r', label='variance confidence intervals')
plt.legend()
plt.title('Model variance depending on model complexity')
plt.show()
