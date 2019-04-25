# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:06:07 2019

@author: pups
"""

import numpy as np
import logging

def svd(X):
    U, S, VT = np.linalg.svd(X)
    assert len(X.shape)==2
    m, n = X.shape
    rank = min(m, n)
    for iter in range(rank):
        a = np.random.randn(n)
        b = np.matmul(X, a) / np.linalg.norm(a)**2
        for i in range(5000):
            a = np.matmul(X.T, b) / np.linalg.norm(b)**2  # (n,)
            b = np.matmul(X, a) / np.linalg.norm(a)**2  # (m,)
            X1 = np.outer(b, a)
            loss = np.linalg.norm(X - X1)**2
            print(i, loss)
    return U, S, VT
