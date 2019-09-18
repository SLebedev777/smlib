# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:32:40 2019

@author: pups
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

from smlib.linear.ols import LinearRegression, bias_variance
from smlib.linear.ridge import Ridge
from smlib.linear.lasso import Lasso
from smlib.linear.lars import LARS

from sklearn.linear_model import LassoLars, lars_path
from sklearn.preprocessing import StandardScaler

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

#boston = datasets.load_boston()
#X, y = boston.data, boston.target

lasso_C = 5  # regularization parameter alpha

X_std = StandardScaler().fit_transform(X)
y_std = StandardScaler(with_std=False).fit_transform(y.reshape(-1, 1)).reshape((len(y)))
alphas, _, sklars_coef_path_ = lars_path(X_std, y_std, method='lasso', verbose=3)

sklasso = LassoLars(alpha=lasso_C, fit_intercept=False)
sklasso.fit(X_std, y_std)

xx = np.sum(np.abs(sklars_coef_path_.T), axis=1)
xx /= xx[-1]

plt.figure(figsize=(12, 8))
plt.plot(xx, sklars_coef_path_.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('sklearn LassoLars Path (weights)')
plt.axis('tight')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(xx, alphas)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Max Covariances = C = alphas')
plt.title('sklearn LassoLars Path (alphas)')
plt.axis('tight')
plt.show()


lasso = Lasso(C=lasso_C)
lasso_w = lasso.fit(X, y)

xx = np.sum(np.abs(lasso.lars.coef_path_), axis=1)
xx_max = xx[-1]
xx /= xx_max

plt.figure(figsize=(12, 8))
plt.plot(xx, lasso.lars.coef_path_)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')

lasso_xx = np.sum(np.abs(lasso_w)) / xx_max
plt.vlines(lasso_xx, ymin, ymax, linestyle='solid', color='r')

plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('smlib LARS(lasso) Path (weights)')
plt.axis('tight')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(xx, lasso.lars.alphas_)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.vlines(lasso_xx, ymin, ymax, linestyle='solid', color='r')
plt.hlines(lasso_C, 0., 1., linestyle='solid', color='r')

plt.xlabel('|coef| / max|coef|')
plt.ylabel('Max Covariances = C = alphas')
plt.title('smlib LARS(lasso) Path (alphas)')
plt.axis('tight')
plt.show()

print(f'sklearn Lasso regularized weights with alpha={lasso_C}: ')
print(sklasso.coef_)
print(f'smlib   Lasso regularized weights with alpha={lasso_C}: ')
print(lasso.coef_)