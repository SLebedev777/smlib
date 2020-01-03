# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:51:54 2019

@author: NRA-LebedevSM
"""

import numpy as np

class RegressionMetrics:
    def __init__(self, y_true, y_pred):
        assert len(y_true) == len(y_pred)
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        self.y_true = y_true
        self.y_pred = y_pred
    
    @property
    def mse(self):
        return np.mean((self.y_true - self.y_pred)**2)
    
    @property
    def mae(self):
        return np.mean(np.abs(self.y_true - self.y_pred))
    
    @property
    def explained_variance(self):
        return 1. - np.var(self.y_true - self.y_pred) / np.var(self.y_true)
    
    @property
    def r2(self):
        return 1. - self.mse / np.var(self.y_true)
    

def aic(y_true, y_pred, k):
    """
    Akaike Information Criterion.
    Designed to compare different linear regression models, depending on their MSE
    on FIXED test set and complexity parameter k (== number of features, if simplified.)
    The lower, the better.
    k* = argmin(aic) for k=[1;K] can be used to feature selection to find optimal
    set of features.
    """
    mse = RegressionMetrics(y_true, y_pred).mse
    N = len(y_true)
    return 2 * k / N + np.log(mse)

def bic(y_true, y_pred, k):
    """
    Bayes Information Criterion.
    Designed to compare different linear regression models, depending on their MSE
    on FIXED test set and complexity parameter k (== number of features, if simplified.)
    The lower, the better.
    k* = argmin(aic) for k=[1;K] can be used to feature selection to find optimal
    set of features.
    """
    mse = RegressionMetrics(y_true, y_pred).mse
    N = len(y_true)
    return k * np.log(N) / N + np.log(mse)


