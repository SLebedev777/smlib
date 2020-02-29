# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:44:06 2020

@author: Семен
"""
import numpy as np
from smlib.decision_trees.dt import DecisionTree
from scipy.stats import mode
from tqdm import tqdm

class RandomForest:
    def __init__(self, task='classification', n_estimators=100, max_depth=10,
                 min_samples_leaf=1, max_features='sqrt',
                 n_jobs=1):
        self.task = task
        if task == 'classification':
            self.criterion = 'gini'
        elif task == 'regression':
            self.criterion = 'mse'
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.was_fit = False
    
    def _create_base_alg(self):
        return DecisionTree(task=self.task, 
                            criterion=self.criterion, 
                            max_depth=self.max_depth,
                            min_samples_leaf=self.min_samples_leaf)
    
    def fit(self, X, y):
        M, N = X.shape
        
        all_train_indices = set(range(M))
        all_features = list(range(N))
        if self.max_features == 'sqrt':
            num_features = int(np.sqrt(N))
        else:
            num_features = N
        
        self.trees = []
        for t in tqdm(range(self.n_estimators)):
            boot_indices = np.unique(np.random.choice(M, M, replace=True))
            boot_size = len(boot_indices)
            feature_indices = np.random.choice(all_features, num_features, replace=False)
            Xt = X[boot_indices, feature_indices].reshape((boot_size, num_features))
            yt = y[boot_indices]
            out_of_bag_indices = sorted(all_train_indices - set(boot_indices))
            X_oob = X[out_of_bag_indices, :]
            y_oob = y[out_of_bag_indices]
            
            tree = self._create_base_alg()
            tree.fit(Xt, yt)
            
            self.trees.append([tree, feature_indices])
        self.was_fit = True
        return self
    
    def predict(self, X):
        M, N = X.shape
        T = len(self.trees)
        
        if not self.was_fit:
            raise ValueError("Estimator was not fit")

        y_pred_matrix = np.zeros((M, T))        
        for t, (tree, feature_indices) in enumerate(self.trees):
            num_features = len(feature_indices)
            Xt = X[:, feature_indices].reshape((M, num_features))
            y_pred_matrix[:, t] = tree.predict(Xt)
        if self.task == 'classification':
            modes, counts = mode(y_pred_matrix, axis=1)
            return modes.reshape(-1)
        if self.task == 'regression':
            return y_pred_matrix.mean(axis=1)
            
