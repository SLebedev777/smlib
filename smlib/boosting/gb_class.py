# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:57:31 2020

@author: Семен
"""
import numpy as np
from smlib.decision_trees.dt import DecisionTree
from functools import partial

gold_phi = 0.5 * (1 + np.sqrt(5))

def golden1dmin(f, a, b, eps):
    if abs(b - a) < eps:
        x1 = 0.5 * (a + b)
        return x1
    t = (b - a) / gold_phi
    x1 = b - t
    x2 = a + t
    y1 = f(x1)
    y2 = f(x2)
    if y1 >= y2:
        return golden1dmin(f, x1, b, eps)
    else:
        return golden1dmin(f, a, x2, eps)


class GBClassifier:
    """
    Gradient Boosting Classification using Decision Trees and Log loss.
    """
    def __init__(self, n_estimators=50, max_depth=1, loss='log', 
                 learning_rate=1.):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        assert loss in ['log', 'exp']
        self.loss_func = loss
        self.algs = []
        self.weights = []       
    
    def _create_base_alg(self):
        return DecisionTree(task='regression', criterion='mse', 
                            max_depth=self.max_depth,
                            min_samples_leaf=1)
    
    def fit(self, X, y):
        assert all (yy in np.array([-1, 1]) for yy in np.unique(y))
        
        def logloss(b, f, h, y):
            return np.mean(np.log(1 + np.exp(-2*y*(f + b*h))))

        def exploss(b, f, h, y):
            return np.mean(np.exp(-2*y*(f + b*h)))
        
        eta = self.learning_rate        
        f = 0.0 # find start approximation
        self.start = f
        for i in range(self.n_estimators):
            # logistic loss (= deviance)
            if self.loss_func == 'log':
                r = -2*y / (1 + np.exp(2*y*f))
            # exponential loss (= AdaBoost)
            elif self.loss_func == 'exp':
                r = -2*y* np.exp(-2*y*f)
                
            base_alg = self._create_base_alg()
            base_alg.fit(X, -r)
            h = base_alg.predict(X)
            loss = partial(logloss, f=f, h=h, y=y)
            b = eta * golden1dmin(loss, 0, 1, eps=1e-3)
            f += b * h
            self.algs.append(base_alg)
            self.weights.append(b)
            print(i, loss(b))
                    
    def decision_function(self, X):
        M, N = X.shape
        y_pred = np.zeros(M) + self.start
        for i in range(self.n_estimators):
            y_pred += self.weights[i] * self.algs[i].predict(X)
        return y_pred
    
    def predict(self, X):
        return np.sign(self.decision_function(X))

