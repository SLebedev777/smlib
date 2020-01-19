#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from tqdm import tqdm

from smlib.utils.one_hot import OneHotEncoder

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


class SupportVectorClassifier:
    """
    SVC for binary classification.
    
    Dual form is solved using scipy.optimize.
    Kernels are supported.
    """
    def __init__(self, C=1., kernel='linear', n_iters=500, tol=1e-3,
                 solver='scipy', verbose=True):
        self.C = C  # regularization coefficient
        assert kernel in ['linear', 'rbf']
        self.kernel = kernel
        self.n_iters = n_iters
        self.tol = tol
        assert solver in ['scipy', 'smo']
        self.solver = solver
        self.verbose = verbose
        self.was_fit = False
    
    def fit(self, X, y):
        assert np.unique(y).tolist() == [-1, 1]
        
        Gram = self._kernel_gram_matrix(X, y)
        if self.solver == 'smo':
            return self._fit_smo(X, y, Gram)
        return self._fit_scipy_dual(X, y, Gram)
        
    def _kernel_gram_matrix(self, X, y):
        M, N = X.shape
        if self.kernel == 'linear':
            self.kernel_func = lambda x1, x2: np.dot(x1, x2.T)
        elif self.kernel == 'rbf':
            self.kernel_func = lambda x1, x2: np.exp(-np.dot(x1-x2, x1-x2)/N)
        else:
            raise ValueError('unknown kernel')

        print('calculating Gram matrix...')
        Gram = np.zeros((M, M))
        for i in tqdm(range(M)):
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

    def _fit_smo(self, X, y, Gram):
        M, N = X.shape
        C = self.C
        alphas = np.zeros(M)
        e = np.ones((M))
        b = 0
        
        def svm_output(point_index, support_indices):
            if len(support_indices) == 0:
                z = 0.
            else:
                z = np.sum([alphas[i]*y[i]*Gram[i, point_index] for i in support_indices])
            return z - b
            
        n_changed = 1
        while(n_changed):
            n_changed = 0        
            for i2 in range(M):
                #print(loss)
                i1 = i2
                while i1 == i2:
                    i1 = np.random.randint(0, M)
                y1 = y[i1]
                y2 = y[i2]
                s = y1 * y2
                support_indices = np.nonzero(alphas)[0]
                E1 = svm_output(i1, support_indices) - y1
                E2 = svm_output(i2, support_indices) - y2
                r2 = E2*y2
                a1 = alphas[i1]
                a2 = alphas[i2]
                if (r2 < -self.tol and a2 < C) or (r2 > self.tol and a2 > 0):
                    if y1 == y2:
                        L = np.max([0., a1 + a2 - C])
                        H = np.min([C, a1 + a2])
                    else:
                        L = np.max([0., a2 - a1])
                        H = np.min([C, a2 - a1 + C])
                    if L == H:
                        continue
                    k11 = Gram[i1, i1]
                    k22 = Gram[i2, i2]
                    k12 = s * Gram[i1, i2]
                    eta = k11 + k22 - 2*k12
                    if eta <= 0:
                        continue
                    a2_new = a2 + y2 * (E1 - E2) / eta
                    a2_new = H if a2_new > H else a2_new
                    a2_new = L if a2_new < L else a2_new
                    a1_new = a1 + s*(a2 - a2_new)
                    alphas[i1] = a1_new
                    alphas[i2] = a2_new
                    b1 = E1 + y1*(a1_new - a1)*k11 + y2*(a2_new - a2)*k12 + b
                    b2 = E2 + y1*(a1_new - a1)*k12 + y2*(a2_new - a2)*k22 + b
                    if 0 < a1_new < C:
                        b = b1
                    elif 0 < a2_new < C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
                    n_changed += 1
            loss = -e.T.dot(alphas) + 0.5 * alphas.T.dot(Gram.dot(alphas))
            print(n_changed, loss)
        
        sv = alphas > 0
        self.support_ = np.where(sv)[0].tolist()
        self.support_vectors_ = X[self.support_, :]
        self.dual_coef_ = alphas[sv] * y[sv]
        self.intercept_ = b
        self.was_fit = True


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.svm import LinearSVC, SVC

    X, y = make_blobs(n_samples=1000, centers=2, random_state=3)
    y[y == 0] = -1
    
    C = 1.
    clf = SupportVectorClassifier(C=C, kernel='linear', solver='smo', verbose=False)
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