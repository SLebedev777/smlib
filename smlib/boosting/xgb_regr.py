# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:57:31 2020

@author: Семен
"""
import numpy as np
from .xgb_rt import XGBRegressionTree


class XGBoostRegressor:
    """
    XGBoost Regressor with Mean Squared Error loss.
    """
    def __init__(self, n_estimators=50, max_depth=4, gamma=0.1,  lambd=1.0,
                 tree_method='hist'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.lambd = lambd
        self.tree_method = tree_method
        self.algs = []
    
    def _create_base_alg(self):
        return XGBRegressionTree(max_depth=self.max_depth,
                                 gamma=self.gamma,
                                 lambd=self.lambd,
                                 tree_method=self.tree_method)
    
    def fit(self, X, y):
        f = 0
        for i in range(self.n_estimators):
            r = y - f
            print(i, np.mean(r**2))
            base_alg = self._create_base_alg()
            base_alg.fit(X, -r)
            f += base_alg.predict(X)
            self.algs.append(base_alg)
                    
    def predict(self, X):
        M, N = X.shape
        y_pred = np.zeros(M)
        for i in range(self.n_estimators):
            y_pred += self.algs[i].predict(X)
        return y_pred
