# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:12:26 2019

@author: pups
"""

import numpy as np
from .ols import LinearRegression, bias_variance

class Ridge(LinearRegression):
    def __init__(self, C=1., intercept=True):
        self.C = C
        super(Ridge, self).__init__(intercept)
        
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        if self.intercept:
            X = self.add_intercept(X)
        X_t = X.transpose()
        cov = np.matmul(X_t, X)
        cov += self.C * np.eye(len(cov))
        self.coef_ = np.matmul(np.linalg.inv(cov), X_t).dot(y).transpose()
        self.rss_ = y - np.dot(X, self.coef_)
        self.fitted = True
        return self.coef_

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

    