# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:37:32 2019

@author: Семен
"""

import numpy as np

from .lars import LARS

class Lasso:
    def __init__(self, C=1., max_steps=500):
        self.fitted = False
        self.C = C
        self.max_steps = max_steps
        self.lars = LARS(method='lasso', max_steps=self.max_steps)
        
    def fit(self, X):
        M, N = X.shape
        assert len(y) == M
        self.coef_ = np.zeros((N,))
        
        self.lars.fit(X, y)

        alphas = self.lars.alphas_.tolist()
        # C is very large -> weights are all zeros
        if self.C >= alphas[0]:
            pass
        # C is close to zero -> weights are full OLS solution
        elif self.C <= alphas[-1]:
            self.coef_ = self.lars.coef_
        # C is somewhere in between 2 LARS steps. Calculate weights proportionally. 
        else:
            C_array = sorted(alphas + [self.C], reverse=True)
            C_index = C_array.index(self.C)
            C1 = C_array[C_index-1]
            C2 = C_array[C_index+1]
            step = (C1 - self.C) / (C1 - C2)
            w1 = self.lars.coef_path_[C_index-1]
            w2 = self.lars.coef_path_[C_index]
            self.coef_ = w1 + step * (w2 - w1)
        
        self.fitted = True    
        return self.coef_
    
    def predict(self, X):
        assert self.fitted
        
        X = self.lars.standardizer_X.transform(X)
        y_pred = np.dot(X, self.coef_)
        y_pred = self.lars.standardizer_y.inverse_transform(y_pred)
        return y_pred

        
    
 