# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

class Standardizer():
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        
    def fit(self, X):
        if self.with_mean:
            self.means = np.mean(X, axis=0)
        else:
            self.means = np.zeros((X.shape[1]))
        if self.with_std:
            self.stds = np.sqrt(np.var(X, axis=0))
        else:
            self.stds = np.ones((X.shape[1]))
        return self

    def transform(self, X):
        assert X.shape[-1] == len(self.means) == len(self.stds)
        X_transformed = np.zeros(X.shape)
        for i in range(X.shape[1]):
            X_transformed[:, i] = (X[:, i] - self.means[i]) / self.stds[i]
        return X_transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        X_inv = np.zeros(X.shape)
        if len(X.shape) == 1:
            for i in range(len(X)):
                X_inv[i] = X[i] * self.stds[0] + self.means[0]
            return X_inv
        if len(X.shape) == 2:
            for i in range(X.shape[1]):
                X_inv[:, i] = X[:, i] * self.stds[i] + self.means[i]
            return X_inv
        else:
            raise ValueError('unsupported shape')

    