# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:40:21 2020

@author: Семен
"""
import numpy as np
from smlib.clustering.gmm import GaussianMixtureEM as GMM


class RBFNetworkClassifier:
    """
    For each class, data distribution is represented as a Gaussian mixture.
    After having learnt per-class GMMs, prediction is done as usual Bayesian classifier.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.was_fit = False
    
    def fit(self, X, y):
        M, N = X.shape
        
        self.classes_labels = np.unique(y)
        K = len(self.classes_labels)
        
        self.classes_log_priors = np.zeros(K)
        self.classes_gmm = [GMM(self.n_components) for i in range(K)]
        for i, c in enumerate(self.classes_labels):
            Xc = X[y == c, :]
            self.classes_gmm[i].fit(Xc)
            self.classes_log_priors[i] = np.log(len(Xc) / len(X))
        
        self.was_fit = True
        return self
            
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
        for i, c in enumerate(self.classes_labels):
            log_pc = self.classes_log_priors[i]
            log_pxy = np.log(self.classes_gmm[i].predict_proba(x.reshape((1, -1))))[0]
            log_pyx.append(log_pc + log_pxy)
        
        log_pyx = np.array(log_pyx)        
        return log_pyx

    def predict_proba(self, X):
        log_pyx = np.apply_along_axis(self._predict_log_pyx_one, axis=1, arr=X)
        pyx = np.exp(log_pyx)
        return np.divide(pyx.T, np.sum(pyx, axis=1)).T
