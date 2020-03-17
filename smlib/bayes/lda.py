# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 00:24:09 2020

@author: Семен
"""

import numpy as np

class LinearDiscriminant:
    """
    Fisher's Linear Discriminant classifier.
    
    Sample covariance matrix is equal for all classes.
    Data is supposed to have multivariate Gaussian distribution within each class.
    """
    def __init__(self):
        self.was_fit = False
    
    def fit(self, X, y):
        M, N = X.shape
        
        self.classes_labels = np.unique(y)
        K = len(self.classes_labels)
        
        X_centered = np.zeros((M, N))
        self.classes_log_priors = np.zeros(K)
        self.mu = np.zeros((K, N))
        for i, c in enumerate(self.classes_labels):
            Xc = X[y == c, :]
            self.mu[i] = np.mean(Xc, axis=0)
            X_centered[y == c, :] = Xc - self.mu[i]
            self.classes_log_priors[i] = np.log(len(Xc) / len(X))
        
        cov = (X_centered.T @ X_centered) / M  # shape is (N, N)
        self.cov_inv = np.linalg.inv(cov)
        self.log_cov_det = np.log(np.linalg.det(cov))
        
        self.was_fit = True
        
            
    def predict(self, X):
        """
        Predict class label according to MAP principle.
        """
        res = np.zeros(len(X))
        log_pyx = np.apply_along_axis(self._predict_log_pyx_one, axis=1, arr=X)
        for i, lpx in enumerate(log_pyx):
            res[i] = self.classes_labels[np.argmax(lpx)]
        return res
    
    def _predict_log_pyx_one(self, x):
        """
        Posterior log proba logp(c|x), according to Bayes rule, is proportional to:
            
            logp(y=k|x) ~~ logp(y=k) + logp(x|y=k) 

        where: 
            p(y=k) is prior for class k
            p(x|y=k) is data likelihood for class k
        
        """
        if not self.was_fit:
            raise ValueError("trying to predict before fit")
        log_pyx = []
        N = x.shape[0]
        constant = - 0.5 * N * np.log(np.pi*2)
        for i, c in enumerate(self.classes_labels):
            log_pc = self.classes_log_priors[i]
            x_mu = x - self.mu[i]
            md_sqr = x_mu @ (self.cov_inv @ x_mu)  # squared Mahalanobis distance
            log_pxy = constant - 0.5 * (self.log_cov_det + md_sqr)
            log_pyx.append(log_pc + log_pxy)
        
        log_pyx = np.array(log_pyx)        
        return log_pyx
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
    
    def predict_proba(self, X):
        log_pyx = np.apply_along_axis(self._predict_log_pyx_one, axis=1, arr=X)
        pyx = np.exp(log_pyx)
        return np.divide(pyx.T, np.sum(pyx, axis=1)).T
    
        
        

        
        
    