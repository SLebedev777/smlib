# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:57:31 2020

@author: Семен
"""
import numpy as np
from smlib.decision_trees.dt import DecisionTree


class GBRegressor:
    """
    Gradient Boosting Regression using Decision Trees and Mean Squared Error loss.
    """
    def __init__(self, n_estimators=50, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.algs = []
        self.weights = []       
    
    def _create_base_alg(self):
        return DecisionTree(task='regression', criterion='mse', 
                            max_depth=self.max_depth,
                            min_samples_leaf=1)
    
    def fit(self, X, y):
        f = np.mean(y)
        self.start = f
        for i in range(self.n_estimators):
            r = y - f
            base_alg = self._create_base_alg()
            base_alg.fit(X, -r)
            h = base_alg.predict(X)
            b = np.dot(r, h) / np.dot(h, h)  # b is just least squares coeff
            f += b * h
            self.algs.append(base_alg)
            self.weights.append(b)
                    
    def predict(self, X):
        M, N = X.shape
        y_pred = np.zeros(M) + self.start
        for i in range(self.n_estimators):
            y_pred += self.weights[i] * self.algs[i].predict(X)
        return y_pred
