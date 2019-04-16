# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:12:26 2019

@author: pups
"""

import numpy as np
from .ols import LinearRegression, bias_variance

class Ridge(LinearRegression):
    def __init__(self, C=1.):
        self.C = C
        super(Ridge, self).__init__(intercept=False)
        
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X_t = X.transpose()
        cov = np.matmul(X_t, X)
        cov += self.C * np.eye(len(cov))
        self.coef_ = np.matmul(np.linalg.inv(cov), X_t).dot(y).transpose()
        self.w0_ = np.mean(y)
        self.residuals_ = y - np.dot(X, self.coef_) - self.w0_
        self.rss_ = np.dot(self.residuals_.T, self.residuals_)
        self.fitted = True
        return self.coef_

    def predict(self, X_pred):
        X_pred = self.scaler.transform(X_pred)
        return np.dot(X_pred, self.coef_) + self.w0_


if __name__ == '__main__': 

    X = np.random.rand(100, 4)
    y = X[:, 2] * 2 + 4
    
    ols = LinearRegression()
    ols.fit(X, y)
    y_pred = ols.predict(X)
    
    print(bias_variance(LinearRegression(), X, y))
    
    ridge = Ridge(.1)
    ridge.fit(X, y)
    
    print(bias_variance(Ridge(.0), X, y))
    print(bias_variance(Ridge(.01), X, y))
    print(bias_variance(Ridge(.1), X, y))
    print(bias_variance(Ridge(1.), X, y))

    