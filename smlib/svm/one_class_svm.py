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


class OneClassSVM:
    """
    One-Class SVM for novelty detection.
    
    Dual form is solved using scipy.optimize.
    Kernels are supported.
    """
    def __init__(self, nu=.1, kernel='rbf', n_iters=500, tol=1e-3, gamma='scale',
                 solver='scipy', verbose=True):
        self.nu = nu  # regularization coefficient
        assert kernel in ['linear', 'rbf']
        self.kernel = kernel
        self.n_iters = n_iters
        self.tol = tol
        assert solver in ['scipy']
        self.solver = solver
        if isinstance(gamma, str):
            assert gamma == 'scale'
        else:
            assert isinstance(gamma, float)
        self.gamma = gamma
        self.verbose = verbose
        self.was_fit = False
    
    def fit(self, X):        
        Gram = self._kernel_gram_matrix(X)
        return self._fit_scipy_dual(X, Gram)
        
    def _kernel_gram_matrix(self, X):
        M, N = X.shape
        if self.kernel == 'linear':
            self.kernel_func = lambda x1, x2: np.dot(x1, x2.T)
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                self.gamma = 1 / N
            self.kernel_func = lambda x1, x2: self.gamma * np.exp(-np.dot(x1-x2, x1-x2))
        else:
            raise ValueError('unknown kernel')

        print('calculating Gram matrix...')
        Gram = np.zeros((M, M))
        for i in tqdm(range(M)):
            for j in range(M):
                Gram[i, j] = self.kernel_func(X[i], X[j])
        return Gram            

    
    def _fit_scipy_dual(self, X, Gram):
        M, N = X.shape
        e = np.ones((M))
        args = (Gram, e)
        
        # Lagrangian for OneClass SVM problem in dual form
        def loss(lambd, *args):
            Gram, e = args
            loss = 0.5 * lambd.T.dot(Gram.dot(lambd))
            return loss
        
        def loss_grad(lambd, *args):
            Gram, e = args
            grad = Gram.dot(lambd)
            return grad
        
        def callback(lambd):
            l = loss(lambd, *args)
            print(f'loss={l:.4f}')

        #  Karush-Kuhn-Tucker constraint for OneClass SVM problem:
        #  np.sum(lambd) = 1
        constraints = [{'type': 'eq',
                        'fun': lambda lambd: np.sum(lambd) - 1,
                        'jac': lambda lambd: e
                        }]
        #  another KKT constraint: 0 <= lambda <= 1/(nu*M)
        B = 1/(self.nu*M)
        bounds = Bounds(0, B)
    
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
        lambd[lambd >= B-1e-5] = B
        
        sv = lambd > 0
                
        # calculate intercept in points that lay exactly on margin
        exact_sv = (0 < lambd) & (lambd < B)
        g = Gram[np.where(sv)[0], :][:, np.where(exact_sv)[0]]
        ro = np.mean(np.dot(lambd[sv], g))
        
        self.support_ = np.where(sv)[0].tolist()
        self.support_vectors_ = X[self.support_, :]
        self.dual_coef_ = lambd[sv]
        self.intercept_ = ro
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
    import matplotlib.font_manager
    from sklearn import svm
    
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate train data
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    
    # fit the model
    models = [OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1),
              svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)]
    for clf in models:
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
        
        # plot the line, the points, and the nearest vectors to the plane
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.title("Novelty Detection")
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
        
        s = 40
        b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
        b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                         edgecolors='k')
        c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                        edgecolors='k')
        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.legend([a.collections[0], b1, b2, c],
                   ["learned frontier", "training observations",
                    "new regular observations", "new abnormal observations"],
                   loc="upper left",
                   prop=matplotlib.font_manager.FontProperties(size=11))
        plt.xlabel(
            "error train: %d/200 ; errors novel regular: %d/40 ; "
            "errors novel abnormal: %d/40"
            % (n_error_train, n_error_test, n_error_outliers))
        plt.show()
        
        print(clf.support_)
        print(clf.dual_coef_)
        print(clf.intercept_)