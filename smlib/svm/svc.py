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
    
    Dual form is solved using scipy.optimize.
    Kernels are supported.
    """
    def __init__(self, C=1., kernel='linear', n_iters=100, tol=1e-3,
                 solver='scipy', verbose=True):
        self.C = C  # regularization coefficient
        assert kernel in ['linear', 'rbf']
        self.kernel = kernel
        self.n_iters = n_iters
        self.tol = tol
        assert solver in ['scipy']
        self.solver = solver
        self.verbose = verbose
        self.was_fit = False
    
    def fit(self, X, y):
        assert np.unique(y).tolist() == [-1, 1]
        
        Gram = self._kernel_gram_matrix(X, y)
        return self._fit_scipy_dual(X, y, Gram)
        
    def _kernel_gram_matrix(self, X, y):
        M, N = X.shape
        if self.kernel == 'linear':
            self.kernel_func = lambda x1, x2: np.dot(x1, x2.T)
        elif self.kernel == 'rbf':
            self.kernel_func = lambda x1, x2: np.exp(-np.dot(x1-x2, x1-x2)/N)
        else:
            raise ValueError('unknown kernel')

        Gram = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                Gram[i, j] = self.kernel_func(X[i], X[j]) * y[i] * y[j]
        return Gram            

    
    def _fit_scipy_dual(self, X, y, Gram):
        M, N = X.shape
        C = self.C
        e = np.ones((M))
        args = (Gram, e)
        
        # Lagrangian for SVM problem in dual form
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

        #  Karush-Kuhn-Tucker constraint for SVM problem:
        #  np.dot(lambd.T, y) = 0
        constraints = [{'type': 'eq',
                        'fun': lambda lambd: np.dot(lambd.T, y),
                        'jac': lambda lambd: y
                        }]
        #  another KKT constraint: 0 <= lambda <= C
        bounds = Bounds(0, C)
    
        print('optimizing by scipy...')
        opt_res = minimize(loss, np.random.rand(M), method='SLSQP',
                     args=args,
                     jac=loss_grad,
                     bounds=bounds,
                     constraints=constraints,
                     callback=callback if self.verbose else None,
                     options={'maxiter': self.n_iters,
                              'disp': self.verbose})
        # found dual coefficients
        lambd = opt_res.x
        # stabilize dual coeffs near box boundaries
        lambd[lambd <= 1e-5] = .0
        lambd[lambd >= C-1e-5] = C
        
        sv = lambd > 0
        num_sv = len(lambd[sv])
        
        self.coef_ = None
        # for linear kernel, calculate weights from primal problem using 
        # dual coefficients from support vectors
        if self.kernel == 'linear':
            self.coef_ = np.sum(np.multiply((y[sv]*lambd[sv]).reshape(num_sv, 1), 
                                            X[sv, :]), axis=0)
        
        # calculate intercept in points that lay exactly on margin boundaries 
        exact_sv = (0 < lambd) & (lambd < C)
        g = Gram[np.where(sv)[0], :][:, np.where(exact_sv)[0]]
        tmp = np.dot(np.multiply(g, y[sv].reshape(num_sv, 1)), lambd[sv])
        w0_ = np.mean(tmp - y[exact_sv])
        
        self.support_ = np.where(sv)[0].tolist()
        self.support_vectors_ = X[self.support_, :]
        self.dual_coef_ = lambd[sv] * y[sv]
        self.intercept_ = w0_
        self.was_fit = True
        return self

    def decision_function(self, X):
        M, N = X.shape
        num_sv = len(self.support_)
        kernel_matrix = np.zeros((num_sv, M))
        for m in range(M):
            for i in range(num_sv):
                kernel_matrix[i, m] = self.kernel_func(self.support_vectors_[i], X[m])
        res = np.dot(self.dual_coef_, kernel_matrix) - self.intercept_
        return res
           
    def predict(self, X):
        return np.sign(self.decision_function(X))


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.svm import LinearSVC, SVC

    X, y = make_blobs(n_samples=40, centers=2, random_state=3)
    y[y == 0] = -1
    
    C = 1
    clf = SupportVectorClassifier(C=C, kernel='rbf', verbose=False)
    #clf = SVC(kernel='rbf', C=C)
    clf.fit(X, y)
    print(clf.support_)
    print(clf.dual_coef_)
    #print(clf.coef_)
    print(clf.intercept_)
        
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    #ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
    #           linewidth=1, facecolors='none', edgecolors='k')
    plt.show()