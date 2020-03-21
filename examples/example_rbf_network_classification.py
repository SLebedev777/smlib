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
from smlib.bayes.rbf_net import RBFNetworkClassifier
from smlib.svm.svc import SupportVectorClassifier


def draw_classes_boundary(ax, X, clf):
    nx, ny = 200, 100
    x_min, x_max = X[:, 0].min(), X[:, 0].max() 
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    points = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(clf, 'predict_proba'):
        Z = clf.predict_proba(points)
        Z = Z[:, 1].reshape(xx.shape)
        levels = [0.5]
    elif hasattr(clf, 'decision_function'):
        Z = clf.decision_function(points).reshape(xx.shape)
        levels = [0]
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.8)
    ax.contour(xx, yy, Z, levels, linewidths=2., colors='r')
    ax.scatter(X[:, 0], X[:, 1], c=y)


n_samples = 150
random_state = 170
X, y = make_blobs(centers=3, n_samples=n_samples, random_state=random_state)
y = np.array([1 if y_==1 else -1 for y_ in y])

for clf_name, clf_type in [
                           ('SVM with RBF kernel',
                            SupportVectorClassifier(kernel='rbf', solver='smo', verbose=False)),
                            ('RBF Network', 
                            RBFNetworkClassifier(2)),
                           ]:
    
    # Correct number of clusters
    rbf1 = clf_type
    rbf1.fit(X, y)
    
    plt.figure(figsize=(15, 15))
    plt.title(clf_name)
    
    ax1 = plt.subplot(221)
    draw_classes_boundary(ax1, X, rbf1)
    
    
    # Anisotropicly distributed data
    transformation = [[0.36, 6.2], [-0.11, 0.05]]
    X_aniso = np.dot(X, transformation)
    
    rbf2 = clf_type
    rbf2.fit(X_aniso, y)
    
    ax2 = plt.subplot(222)
    draw_classes_boundary(ax2, X_aniso, rbf2)
    
    
    # Different variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
    y_varied = np.array([1 if y_==1 else -1 for y_ in y_varied])
    
    rbf3 = clf_type
    rbf3.fit(X_varied, y_varied)
    
    ax3 = plt.subplot(223)
    draw_classes_boundary(ax3, X_varied, rbf3)
    
    plt.show()
