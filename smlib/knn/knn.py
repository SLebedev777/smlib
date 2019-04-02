# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:54:38 2019

@author: NRA-LebedevSM
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

class kNN:
    """
    Nearest Neighbors model.
    Input data must be numeric only (not categorical).
    All features will be standartized to mean=0 and variance=1.
    """
    def __init__(self, task='regression', k=5, metric='l2'):
        self.task = task
        self.k = k
        if metric == 'l2':
            self.metric_func = lambda x: np.linalg.norm(x, axis=1)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        self.X = self.scaler.fit_transform(X)
        self.y = y
        return self

    def predict(self, X_pred):
        X_pred = self.scaler.transform(X_pred)
        indices = []
        for xp in X_pred:
            distances = self.metric_func(self.X - xp)
            indices.append(np.argsort(distances)[:self.k])
        if self.task == 'regression':
            return np.array([np.mean(self.y[i]) for i in indices])
        if self.task == 'classification':
            return np.array([Counter(self.y[i]).most_common(1) for i in indices])
        
    
        
                
            
            
            