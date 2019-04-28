# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:06:07 2019

@author: pups
"""

import numpy as np
import logging
from numpy.linalg import norm

def svd(X):
    return np.linalg.svd(X, full_matrices=False)

def my_svd(X):
    assert len(X.shape)==2
    m, n = X.shape
    rank = min(m, n)
    singulars = []
    a_vectors = []
    b_vectors = []
    P = np.zeros(X.shape)
    for iter in range(min(m, n)):
        X -= P
        a = np.random.rand(n)
        a /= np.sum(a**2)
        b = np.matmul(X, a) / np.sum(a**2)
        for i in range(50):
            a = np.matmul(X.T, b) / np.sum(b**2)  # (n,)
            b = np.matmul(X, a) / np.sum(a**2)  # (m,)
        P = np.outer(b, a)
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        s = an * bn
        singulars.append(s)
        a_vector = a / an
        b_vector = b / bn
        a_vectors.append(a / an)
        b_vectors.append(b / bn)
        #print('-'*50)
        #print('iteration:', iter)
        #print('s: ', s)
        #print('b_vector  (left): ', b_vector)
        #print('a_vector (right): ', a_vector)
    VT = np.matrix(a_vectors[::-1])
    U  = np.matrix(b_vectors[::-1]).T
    S  = np.array(singulars[::-1])
    return U, S, VT
'''
X = np.random.rand(400, 200)
X -= X.mean()

U, S, VT = my_svd(X)
X_reconstruct = np.dot(U[:, 0] * S[0], VT[0, :])
print(np.allclose(X, X_reconstruct))

U1, S1, VT1 = svd(X)
X_reconstruct1 = np.outer(U1[:, 0] * S1[0], VT1[0, :])
print(np.allclose(X, X_reconstruct1))
'''

def eigen(X, max_iters=10, eps=1e-07):
    # rayleigh iterative algorithm
    assert len(X.shape) == 2
    m, n = X.shape
    assert m == n
    assert np.allclose(X, X.T)  # check symmetric, to have real eigenvalues
    eigenpairs = []
    for j in range(n):
        u = np.random.rand(n)
        u[j] = 1  # ensure to have non-zero start for j-th eigen direction
        for i in range(max_iters):
            u1 = X @ u
            mu = norm(u1) / norm(u)
            r = u1.T @ (X @ u1) / (u1.T @ u1)
            u = u1
        u /= norm(u)
        eigenpairs.append([r, u])
        p = np.nonzero(u)[0]
        defl = (u @ X[p]) / norm(u)
        X = X - defl
        #x*transpose(x)*A*x*transpose(x)
    return eigenpairs

X = np.diag([30, 50, 200, 1])
print(eigen(X))