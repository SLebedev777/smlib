# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:08:31 2020

@author: Семен
"""
from smlib.clustering.hierarchy import AgglomerativeClustering
from smlib.dim_reduction.pca import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram

iris = load_iris()
X = iris.data

model_params = {'affinity': 'euclidean', 'linkage': 'single',
                'cut_mode': 'dist_thres', 'cut_value': 1}    
model = AgglomerativeClustering(**model_params)
model.fit(X)

plt.figure(figsize=(15, 8))
dendrogram(model.Z_, truncate_mode='level', p=0)
plt.title("Dendrogram of HAC on Iris data")
plt.show()


X_2d = PCA().fit_transform(X)

plt.figure(figsize=(16, 8))
ax1 = plt.subplot(121)
ax1.title.set_text('PCA 2d projection of Iris data with true classes')
ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=iris.target)

ax2 = plt.subplot(122)
ax2.title.set_text('PCA 2d projection of Iris data after HAC')
ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=model.labels_)