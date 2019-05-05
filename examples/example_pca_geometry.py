# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:04:05 2019

@author: pups
"""
import numpy as np
from sklearn.datasets import make_blobs
from smlib.core.linalg import svd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as skPCA
from smlib.dim_reduction.pca import PCA as smPCA

X, y = make_blobs(1000, 2, centers=[[0, 0]])
transformation = [[0.4, 1.5], [-2.5, 3.2]]
X = np.dot(X, transformation)

U, S, VT = svd(X)
pca_components = VT.T

origin = [0], [0]
plt.figure(figsize=(7, 7))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(X[:, 0], X[:, 1])
plt.quiver(*origin, pca_components[0, :], pca_components[1, :])

sk_pca = skPCA(2)
sk_pca.fit(X)

sm_pca = smPCA(2)
sm_pca.fit(X)

print(np.allclose(np.abs(sk_pca.transform(X)), np.abs(sm_pca.transform(X))))