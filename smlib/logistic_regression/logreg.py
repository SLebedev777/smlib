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

from smlib.utils.one_hot import OneHotEncoder

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

def sigmoid(z):
    return 1./(1. + np.exp(-z))


def softmax(z):
    """
    z is vector of scalars (usually dots between X and Wi).
    """
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


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


class MulticlassLogisticRegression:
    """
    Logistic Regression for multiclass classification.
    L2-regularization is supported.
    
    """
    def __init__(self, fit_intercept=True, C=1., alpha=.01, n_iters=100, tol=1e-6,
                 solver='gd', label_encoder=OneHotEncoder()):
        self.fit_intercept = fit_intercept
        self.C = C  # regularization coefficient
        self.alpha = alpha  # gradient step coeff
        self.n_iters = n_iters
        self.tol = tol
        assert solver in ['gd']
        self.solver = solver
        self.label_encoder = label_encoder
        self.was_fit = False
    
    def fit(self, X, y):
        if self.label_encoder:
            y = self.label_encoder.fit_transform(y)
            
        return self._fit_gd(X, y)

    def _fit_gd(self, X, y):
        M, N = X.shape
        _, K = y.shape
        
        w = np.zeros((N, K))
        w0 = np.zeros(K)
        XT = X.T
        print('before start', self.logloss(X, w, w0, y))
        for ni in range(self.n_iters):
            p = self._pp(X, w, w0)
            grad = np.dot(XT, p - y) + self.C * w
            step = self.alpha * grad / M
            w -= step
            if self.fit_intercept:
                grad_w0 = np.sum(p - y, axis=0)
                step_w0 = self.alpha * grad_w0 / M
                w0 -= step_w0
            if np.linalg.norm(step) < self.tol:
                break
            print(ni, self.logloss(X, w, w0, y))
        self.coef_ = w
        self.intercept_ = w0
        self.was_fit = True
        return self
        
    
    def _pp(self, X, w, w0):
        # internal method for proba calculation (forward step)
        z = np.dot(X, w) + w0  # TODO: check for multiclass
        p = np.zeros(z.shape)
        for i, row in enumerate(z):
            p[i] = softmax(row)  # matrix of probas, shape (len(X), num_classes)
        return p
        
    def logloss(self, X, w, w0, y):
        M = X.shape[0]
        p = self._pp(X, w, w0)
        loss = -np.sum(np.sum(y*np.log(p), axis=1))
        reg = 0.5 * self.C * np.sum(w**2)  # don't penalize intercept weight

        return (loss + reg) / M

    def predict_proba(self, X):
        if not self.was_fit:
            raise ValueError("trying to predict before fit")
        return self._pp(X, self.coef_, self.intercept_)

    def predict(self, X):
        if not self.was_fit:
            raise ValueError("trying to predict before fit")
        p = self.predict_proba(X)
        predictions = np.array([np.argmax(row) for row in p])
        y_pred = np.zeros(p.shape)
        for i, r in enumerate(predictions):
            y_pred[i, r] = 1
        if self.label_encoder:
            y_pred = self.label_encoder.inverse_transform(y_pred)
        return y_pred
