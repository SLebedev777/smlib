# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:39:25 2020

@author: Семен
"""
import numpy as np

def sigmoid(z):
    return 1./(1. + np.exp(-z))


class SGDClassifier:
    def __init__(self, loss='hinge', max_iter=1000, tol=0.001, 
                 eta0=0.001, alpha=1):
        assert loss in ['hinge', 'log', 'perceptron']
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.alpha = alpha
    
    def fit(self, X, y):
        M, N = X.shape
        
        # make target class {+1; -1}
        ys = np.array([1 if p == 1 else -1 for p in y])
        
        w = np.random.randn(N)
        b = 0.
        
        eta = self.eta0


        def out(i):
            return ys[i] * (np.dot(X[i, :], w) - b)
        
        # SVM
        def svm_L():
            outs = (np.dot(X, w) - b) * ys
            hinge = [0 if m >= 1. else 1 - m for m in outs]
            return (self.alpha / 2) * np.dot(w, w) + np.mean(hinge)

        def svm_dLdw(i, margin):
            hinge_grad = -X[i, :] * ys[i] if margin < 1.0 else 0.0
            return hinge_grad + self.alpha * w
        
        def svm_dLdb(i, margin):
            return ys[i] if margin < 1.0 else 0.0

        # LR
        def log_L():
            outs = (np.dot(X, w) - b) * ys
            return (self.alpha / 2) * np.dot(w, w) + np.mean(np.log2(1. + np.exp(-outs)))
        
        def log_dLdw(i, margin):
            return -sigmoid(-margin) * X[i, :] * ys[i] + self.alpha * w
        
        def log_dLdb(i, margin):
            return sigmoid(-margin)

        # perceptron
        def perc_L():
            outs = (np.dot(X, w) - b) * ys
            return np.mean([0 if m >= 0 else -m for m in outs])
        
        def perc_dLdw(i, margin):
            return -X[i, :] * ys[i] if margin < 0.0 else 0.0
        
        def perc_dLdb(i, margin):
            return ys[i] if margin < 0.0 else 0.0

        if self.loss == 'hinge':
            L = svm_L
            dLdw = svm_dLdw
            dLdb = svm_dLdb
        if self.loss == 'log':
            L = log_L
            dLdw = log_dLdw
            dLdb = log_dLdb
        if self.loss == 'perceptron':
            L = perc_L
            dLdw = perc_dLdw
            dLdb = perc_dLdb
        
        
        prev_loss = L()        
        for it in range(self.max_iter):
            for i in np.random.choice(M, size=M, replace=False):
                margin = out(i)
                w_step = dLdw(i, margin)
                b_step = dLdb(i, margin)
                w -= eta*w_step
                b -= eta*b_step
            if it % 100 == 0:
                curr_loss = L()
                print(it, curr_loss)
                if np.abs(curr_loss - prev_loss) <= self.tol:
                    break
                prev_loss = curr_loss
        
        self.coef_ = w
        self.intercept_ = b

    def decision_function(self, X):
        return np.dot(X, self.coef_) - self.intercept_
    
    def predict(self, X):
        return np.sign(self.decision_function(X))
