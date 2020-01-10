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
        e = np.ones((M))
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
                     callback=callback,
                     options={'maxiter': self.n_iters,
                              'disp': True})
        lambd = opt_res.x
        lambd[lambd <= 1e-5] = .0
        
        sv = lambd > 0
        num_sv = len(lambd[sv])
        w = np.sum(np.multiply((y[sv]*lambd[sv]).reshape(num_sv, 1), X[sv, :]), axis=0)
        
        exact_sv = (0 < lambd) & (lambd < C)
        w0 = np.mean(np.dot(X[exact_sv, :], w) - y[exact_sv])
        
        self.support_ = np.where(sv)[0].tolist()
        self.coef_ = w
        self.dual_coef_ = lambd[sv]
        self.intercept_ = w0
        self.was_fit = True
        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_) - self.intercept_
           
    def predict(self, X):
        return np.sign(self.decision_function(X))


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.svm import LinearSVC, SVC

    X, y = make_blobs(n_samples=800, centers=2, random_state=50)
    y[y == 0] = -1
    
    #clf = SupportVectorClassifier(C=0.001)
    clf = SVC(kernel='linear', C=0.001)
    clf.fit(X, y)
    print(clf.support_)
    print(clf.dual_coef_)
    print(clf.coef_)
    print(clf.intercept_)
    
    
    #svc = SVC(kernel='linear')
    #svc.fit(X, y)
    #print(svc.support_)
    #print(svc.dual_coef_)
    #print(svc.coef_)
    #print(svc.intercept_)
    
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