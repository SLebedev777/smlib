# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:06:07 2019

@author: pups
"""

import numpy as np
import logging
from numpy.linalg import norm


def eigen(X, max_iters=70):
    # rayleigh iterative algorithm with simple deflation
    assert len(X.shape) == 2
    m, n = X.shape
    assert m == n
    assert np.allclose(X, X.T)  # check symmetric, to have real positive eigenvalues
    eigenvalues  = []
    eigenvectors = []
    for j in range(n):
        u = np.random.rand(n)
        u[j] = 1  # ensure to have non-zero start for j-th eigen direction
        for i in range(max_iters):
            u1 = X @ u  # just primitive power method
            mu = u1.T @ (X @ u1) / (u1.T @ u1)  # rayleigh estimate of eigenvalue
            u = u1
        u /= norm(u)
        defl = mu * np.outer(u, u.T)
        X = X - defl
        eigenvalues.append(mu)
        eigenvectors.append(u)
    return np.array(eigenvalues), np.array(eigenvectors)

def svd(X):
    # naive and inefficient SVD algorithm. Use for study purpose only.
    assert len(X.shape)==2
    cov = X.T @ X
    eigenvalues, eigenvectors = eigen(cov / norm(cov))
    eigenvalues *= norm(cov)
    S = eigenvalues**0.5
    VT = eigenvectors
    U = X @ (VT.T @ np.diag(1 / S))
    return U, S, VT


if  __name__ == '__main__':    
    print('compare Numpy and Smlib eigen calculations')
    X = np.random.rand(1000, 100)
    cov = (X.T @ X) / len(X)
    eigenvalues, eigenvectors = eigen(cov)
    np_evals, np_evects = np.linalg.eig(cov)
    print(norm(np.sort(np_evals) - np.sort(eigenvalues)))
    
    print('check eigen decomposition')
    X = np.diag([200, 50, 30, 29])
    eigenvalues, eigenvectors = eigen(X)
    print(eigenvectors)
    V = eigenvectors.T
    X_hat = V @ (np.diag(eigenvalues) @ V.T)  # reconstruct X
    print(X_hat.round(3))
    
    print('check SVD')
    X = np.random.rand(10000, 50)
    U, S, VT = svd(X)
    X_hat = U @ (np.diag(S) @ VT)
    print(np.allclose(X, X_hat))
    print(norm(X - X_hat))