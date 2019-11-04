# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 00:55:51 2019

@author: Семен
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
from smlib.clustering.gmm import GaussianMixtureEM as GMM

# helper function that draws ellipses around 2D Gaussian mixtures that have
# diagonal covariance matrices
def draw_gaussian_3sigms_2d(ax, mu, cov):
    # mu.shape = (K, 2)
    # cov.shape = (K, 2)
    K = mu.shape[0]
    for k in range(K):
        sigmas = [np.sqrt(cov[k][i, i]) for i in range(len(cov[k]))]
        ell = Ellipse(mu[k], 6*sigmas[0], 6*sigmas[1])
        ell.set_alpha(0.2)
        ax.add_patch(ell)


n_samples = 1500
random_state = 170
X, y = make_blobs(centers=3, n_samples=n_samples, random_state=random_state)

# Correct number of clusters
gmm1 = GMM(n_components=3, random_state=random_state)
y_pred = gmm1.fit_predict(X)

plt.figure(figsize=(15, 15))

ax1 = plt.subplot(221)    
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
draw_gaussian_3sigms_2d(ax1, gmm1.mu_, gmm1.cov_)
    
# Anisotropicly distributed data
transformation = [[0.36, 6.2], [-0.11, 0.05]]
X_aniso = np.dot(X, transformation)
gmm2 = GMM(n_components=3, random_state=random_state)
y_pred = gmm2.fit_predict(X_aniso)

ax2 = plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
draw_gaussian_3sigms_2d(ax2, gmm2.mu_, gmm2.cov_)
plt.title("Anisotropicly Distributed Blobs")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
gmm3 = GMM(n_components=3, random_state=random_state)
y_pred = gmm3.fit_predict(X_varied)

ax3 = plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
draw_gaussian_3sigms_2d(ax3, gmm3.mu_, gmm3.cov_)
plt.title("Unequal Variance")

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
gmm4 = GMM(n_components=3, random_state=random_state)
y_pred = gmm4.fit_predict(X_filtered)

ax4 = plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
draw_gaussian_3sigms_2d(ax4, gmm4.mu_, gmm4.cov_)
plt.title("Unevenly Sized Blobs")

plt.show()
