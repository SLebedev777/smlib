# -*- coding: utf-8 -*-
"""
Created on Fri May  3 23:45:29 2019

@author: pups
"""

import numpy as np
from smlib.core.linalg import svd

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
    
    def fit(self, X):
        assert len(X.shape) == 2
        m, n = X.shape
        X = X - X.mean(axis=0)  # center features to zero
        U, S, VT = svd(X)
        k = n if self.n_components is None else self.n_components
        self.n_components = k
        self.singular_values = S[:k]
        self.components = VT[:, :k]
        self.explained_variance = (S**2) / len(X)
        total_variance = np.sum(self.explained_variance)
        self.explained_variance_ratio = self.explained_variance / total_variance
        self.explained_variance = self.explained_variance[:k]
        self.explained_variance_ratio = self.explained_variance_ratio[:k]
    
    def transform(self, X):
        assert len(X.shape) == 2
        X_mean = X.mean(axis=0)
        X = X - X_mean
        X_trans = np.dot(X, self.components.T)
        return X_trans
        
    
        
            