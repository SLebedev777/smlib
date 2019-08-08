#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:29:22 2019

@author: sl
"""

import logging
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

def sigmoid(z):
    return 1./(1. + np.exp(-z))


class LogisticRegression:
    """
    Logistic Regression for binary classification.
    L2-regularization is supported.
    
    """
    def __init__(self, fit_intercept=True, C=1., alpha=1., n_iters=100, tol=1e-6,
                 solver='gd'):
        self.fit_intercept = fit_intercept
        self.C = C  # regularization coefficient
        self.alpha = alpha  # gradient step coeff
        self.n_iters = n_iters
        self.tol = tol
        assert solver in ['gd', 'newton-cg']
        self.solver = solver
        self.was_fit = False
    
    def fit(self, X, y):
        if self.solver == 'gd':
            return self._fit_gd(X, y)
        if self.solver == 'newton-cg':
            try:
                return self._fit_newton_cg(X, y)
            except Exception as e:
                print('failed to start newton-cg, reason: ', str(e))
                print('falling back to GD')
                return self._fit_gd(X, y)
        else:
            return self._fit_gd(X, y)
        
    
    def _fit_gd(self, X, y):
        M, N = X.shape
        w = np.zeros((N))
        w0 = 0  # starting weight for intercept
        XT = X.T
        for ni in range(self.n_iters):
            z = np.dot(X, w) + w0
            p = sigmoid(z)
            grad = np.dot(XT, p - y) + self.C * w
            step = self.alpha * grad / M
            w -= step
            if self.fit_intercept:
                grad_w0 = np.sum(p - y)
                step_w0 = self.alpha * grad_w0 / M
                w0 -= step_w0
            if np.linalg.norm(step) < self.tol:
                break
            print(ni)
        self.coef_ = w
        self.intercept_ = w0
        self.was_fit = True
        return self

    
    def _fit_newton_cg(self, X, y):
        M, N = X.shape
        C = self.C
        if self.fit_intercept:
            X = np.hstack([np.ones((M, 1)), X])
            N += 1
        w = np.zeros((N))
        args = (X, y, C, self.fit_intercept)
        
        def logloss(w, *args):
            X, y, C, fit_intercept = args
            M, N = X.shape
            z = np.dot(X, w)
            loss = -( np.dot(y.T, z) - np.sum(np.log(1. + np.exp(z))) )
            if fit_intercept:
                reg = 0.5 * C * np.sum(w[1:]**2)
            else:
                reg = 0.5 * C * np.sum(w**2)  # don't penalize intercept weight
            return (loss + reg) / M
        
        def logloss_grad(w, *args):
            X, y, C, fit_intercept= args
            M, N = X.shape
            z = np.dot(X, w)
            p = sigmoid(z)
            grad = np.dot(X.T, p - y) + C * w
            if fit_intercept:
                grad[0] = np.sum(p - y)  # don't penalize intercept weight
            return grad / M
        
        def logloss_hess(w, *args):
            X, y, C, fit_intercept = args
            M, N = X.shape
            z = np.dot(X, w)
            p = sigmoid(z)
            r = np.multiply(p, 1-p).reshape((M, 1))            
            hessian = np.dot(X.T, np.multiply(r, X))
            start = 0
            if fit_intercept:
                hessian[0, :] = np.zeros((N))
                hessian[:, 0] = np.zeros((N))
                hessian[0, 0] = 1.
                start = 1  # don't penalize intercept weight
            for i in range(start, N):
                hessian[i, i] += C
            return hessian

        def callback(w):
            loss = logloss(w, *args)
            print(f'loss={loss}')

        
        print('optimizing by newton-cg...')
        opt_res = minimize(logloss, w, method='Newton-CG',
                     args=args,
                     jac=logloss_grad, 
                     hess=logloss_hess,
                     callback=callback, 
                     options={'xtol': self.tol, 'maxiter': self.n_iters,
                              'disp': True})
        
        if self.fit_intercept:
            self.coef_ = opt_res.x[1:]
            self.intercept_ = opt_res.x[0]
        else:
            self.coef_ = opt_res.x
            self.intercept_ = 0
        self.was_fit = True
        return self


            
    def predict_proba(self, X):
        if not self.was_fit:
            raise ValueError("trying to predict before fit")
        z = np.dot(X, self.coef_) + self.intercept_
        p = sigmoid(z)
        return p

    def predict(self, X):
        if not self.was_fit:
            raise ValueError("trying to predict before fit")
        z = np.dot(X, self.coef_) + self.intercept_
        p = sigmoid(z)
        return np.array([1 if pr>0.5 else 0 for pr in p])

