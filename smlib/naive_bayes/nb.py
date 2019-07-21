#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:29:22 2019

@author: sl
"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

class GaussianNB:
    """
    Gaussian Naive Bayes classifier.
    
    Features are supposed to be real-valued, normally distributed within 
    each class.
    """
    def __init__(self):
        self.C = 0  # number of classes
        self.N = 0  # number of features 
        self.classes_probas = None  # classes prior probas
        self.classes_labels = None
        self.mus = None  # matrix of Gaussian mean for every class and feature
        self.sigmas = None # matrix of Gausisan std for every class and feature
        self.constants = None # matrix of constants in Gaussian depending only by std
        self.was_fit = False
    
    def fit(self, X, y):
        # for fit(), X and y are specified in sklearn fashion:
        # X is numpy array of shape (n_samples, n_features)
        # y is numpy array of shape (n_samples)
        dfx = pd.DataFrame(X)
        dfy = pd.Series(y)
        self.classes_probas = (dfy.value_counts() / len(y)).sort_index()
        self.classes_labels = self.classes_probas.index.values.tolist()
        self.C = len(self.classes_labels)
        self.N = X.shape[1]
        self.mus = np.zeros((self.C, self.N))
        self.sigmas = np.zeros((self.C, self.N))
        self.constants = np.zeros((self.C, self.N))
        # calculate small additive fraction of most wide std from features.
        # it's needed to avoid numerical unstability,
        # when sigma estimate from feature std is too close to 0
        self.std_eps = 1e-4*np.std(X, axis=0).max()
        for i, c in enumerate(self.classes_labels):
            dfxc = dfx[dfy==c]
            means = dfxc.apply(lambda f: np.mean(f.values, dtype=np.float64), axis=0).values
            stds  = dfxc.apply(lambda f: np.std(f.values, dtype=np.float64), axis=0).values
            stds += self.std_eps
            constants = -0.5*np.log(2.*np.pi) - np.log(stds)
            self.mus[i] = means
            self.sigmas[i] = stds
            self.constants[i] = constants
        self.was_fit = True
        return self
    
    def _predict_one(self, x):
        """
        Predict class label given new x, using MAP rule on posterior log probas.
        Posterior log proba logp(c|x) is calculated from Bayes rule:
            
            logp(y|x) ~~ logp(y) + sum( logp(x|y) ) 
            
        for GaussianNB,  p(x|y) ~ N(mu, std)
            
        Predicted class label is c = argmax logp(y|x), for y in classes_labels
        """
        if not self.was_fit:
            raise ValueError("trying to predict before fit")
        log_pyx = []
        for i, c in enumerate(self.classes_labels):
            log_pc = np.log(self.classes_probas[c])
            log_pxy = np.array([self.constants[i, f] - \
                                (0.5*(x[f] - self.mus[i, f])**2)/(self.sigmas[i, f]**2) \
                                for f in range(self.N)])
            log_pyx.append([log_pc + np.sum(log_pxy), c])
            
        log_pyx = sorted(log_pyx, key=lambda item: item[0], reverse=True)
        return log_pyx[0][1]
            
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])


    
