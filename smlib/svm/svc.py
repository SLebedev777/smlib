#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

from smlib.utils.one_hot import OneHotEncoder

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


class SupportVectorClassifier:
    """
    SVC for binary classification.
    
    """
    def __init__(self, C=1., kernel='linear', n_iters=100, tol=1e-3,
                 solver='scipy'):
        self.C = C  # regularization coefficient
        assert kernel in ['linear']
        self.kernel = kernel
        self.n_iters = n_iters
        self.tol = tol
        assert solver in ['scipy']
        self.solver = solver
        self.was_fit = False
    
    def fit(self, X, y):
        assert np.unique(y).tolist() == [-1, 1]
        
        Gram = self._kernel_gram_matrix(X, y)
        return self._fit_scipy_dual(X, y, Gram)
        
    def _kernel_gram_matrix(self, X, y):
        M, N = X.shape
        if self.kernel == 'linear':
            tmp = np.multiply(X, y.reshape(M, 1))
            return np.dot(tmp, tmp.T)
        else:
            raise ValueError('unknown kernel')
    
    def _fit_scipy_dual(self, X, y, Gram):
        M, N = X.shape
        C = self.C
        w = np.zeros((N))
        e = np.ones((M))
        lambd = np.zeros((M))
        args = (Gram, e)
        
        def loss(lambd, *args):
            Gram, e = args
            loss = -e.T.dot(lambd) + 0.5 * lambd.T.dot(Gram.dot(lambd))
            return loss
        
        def loss_grad(lambd, *args):
            Gram, e = args
            grad = -e + Gram.dot(lambd)
            return grad
        
        def callback(lambd):
            l = loss(lambd, *args)
            print(f'loss={l:.4f}')

        def kkt(lambd, *args):
            """
            Karush-Kuhn-Tucker constraint for SVM problem:
                np.dot(lambd.T, y) = 0
            """
            y = args[0]
            return np.dot(lambd.T, y)
        
        constraints = [{'type': 'eq',
                        'fun': lambda lambd: np.dot(lambd.T, y),
                        'jac': lambda lambd: y
                        #'fun': kkt,
                        #'args': [y,]
                }
                ]
        
        print('optimizing by scipy...')
        opt_res = minimize(loss, np.random.rand(M), method='SLSQP',
                     args=args,
                     jac=loss_grad,
                     bounds=Bounds(0, C),  # another KKT equality: 0 <= lamb
                     constraints=constraints,
                     callback=callback,
                     options={'maxiter': self.n_iters,
                              'disp': True})
        
        self.dual_coef_ = opt_res.x
        #self.coef_ = opt_res.x
        self.was_fit = True
        return self
            


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.svm import LinearSVC, SVC

    X, y = make_blobs(n_samples=800, centers=2, random_state=10)
    y[y == 0] = -1
    
    clf = SupportVectorClassifier()
    clf.fit(X, y)
    print(clf.dual_coef_[clf.dual_coef_ > 1e-3])
    
    svc = SVC(kernel='linear')
    svc.fit(X, y)
    print(svc.support_)
    print(svc.dual_coef_)