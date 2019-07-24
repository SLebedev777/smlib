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
        self.classes_log_probas = None  # classes prior log probas
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
        self.classes_log_probas = np.log((dfy.value_counts() / len(y)).sort_index())
        self.classes_labels = self.classes_log_probas.index.values.tolist()
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
            means = dfxc.apply(lambda feature: 
                np.mean(feature.values, dtype=np.float64), axis=0).values
            stds  = dfxc.apply(lambda feature: 
                np.std(feature.values, dtype=np.float64), axis=0).values
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
            log_pc = self.classes_log_probas[c]
            log_pxy = self.constants[i] - (0.5*(x - self.mus[i])**2) / (self.sigmas[i]**2)
            log_pyx.append(log_pc + np.sum(log_pxy))
            
        return self.classes_labels[np.argmax(np.array(log_pyx))]
            
    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis=1, arr=X)



class BernoulliNB:
    """
    Bernoulli Naive Bayes classifier.
    
    Features are supposed to be boolean. 
    If feature values are non-negative real values, they are transformed to boolean.
    
    Only dense numpy arrays are supported.
    """
    def __init__(self):
        self.C = 0  # number of classes
        self.N = 0  # number of features 
        self.classes_log_probas = None  # classes prior log probas
        self.classes_labels = None
        self.logp = None  # matrix of Bernoulli log probas for every class and feature
        self.neglogp = None  # matrix for log(1 - p)
        self.was_fit = False
    
    def fit(self, X, y):
        # for fit(), X and y are specified in sklearn fashion:
        # X is numpy array of shape (n_samples, n_features)
        # y is numpy array of shape (n_samples)
        assert np.all(X >= 0.)
        dfx = pd.DataFrame(X).astype(bool).astype(int)
        dfy = pd.Series(y)
        self.classes_log_probas = np.log((dfy.value_counts() / len(y)).sort_index())
        self.classes_labels = self.classes_log_probas.index.values.tolist()
        self.C = len(self.classes_labels)
        self.N = X.shape[1]
        self.logp = np.zeros((self.C, self.N))
        self.neglogp = np.zeros((self.C, self.N))
        for i, c in enumerate(self.classes_labels):
            dfxc = dfx[dfy==c]
            p = dfxc.apply(lambda feature: 
                (1 + np.count_nonzero(feature.values)) / (2+ len(feature)), axis=0).values
            self.logp[i] = np.log(p)
            self.neglogp[i] = np.log(1. - p)
        self.was_fit = True
        return self
    
    def _predict_one(self, x):
        """
        Predict class label given new x, using MAP rule on posterior log probas.
        Posterior log proba logp(c|x) is calculated from Bayes rule:
            
            logp(y|x) ~~ logp(y) + sum( logp(x|y) ) 
            
        for BernoulliNB,  logp(x|yi) = xi * log(Pi) + (1 - xi) * log(1 - Pi) 
            
        Predicted class label is c = argmax logp(y|x), for y in classes_labels
        """
        if not self.was_fit:
            raise ValueError("trying to predict before fit")
        log_pyx = []
        for i, c in enumerate(self.classes_labels):
            log_pc = self.classes_log_probas[c]
            log_pxy = x * self.logp[i] + self.neglogp[i] - x * self.neglogp[i]
            log_pyx.append(log_pc + np.sum(log_pxy))
        
        return self.classes_labels[np.argmax(np.array(log_pyx))]
            
    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis=1, arr=X)



    
