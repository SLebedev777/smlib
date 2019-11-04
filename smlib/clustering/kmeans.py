# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:26:56 2019

@author: Семен
"""
import numpy as np


class KMeans:
    def __init__(self, n_clusters, eps=1e-4, n_init=10, max_iter=300, random_state=None):
        self.num_clusters = n_clusters
        self.eps = eps
        self.n_init = n_init
        self.max_iter = max_iter
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif type(random_state) == int:
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        self.was_fit = False

   
    def fit(self, X):
        M, N = X.shape
        K = self.num_clusters
        best_inertia = 0

        for attempt in range(self.n_init):
            
            centroids = X[self.random_state.choice(range(M), size=K), :]  # (K, N)
            prev_inertia = 0
            for it in range(self.max_iter):
                
                # E step
                cluster_labels, inertia = self._e_step(X, centroids)

                if prev_inertia and (prev_inertia - inertia <= self.eps):
                    break
                prev_inertia = inertia
    
                # M step
                for c in range(K):
                    new_centroid = np.mean(X[cluster_labels == c], axis=0)  # (1, N)
                    centroids[c] = new_centroid
            
            if attempt == 0 or (best_inertia and inertia < best_inertia):
                best_inertia = inertia
                best_centroids = centroids
                best_cluster_labels = cluster_labels

        self.centroids = best_centroids
        self.cluster_labels = best_cluster_labels
        self.was_fit = True


    def fit_predict(self, X):
        self.fit(X)        
        return self.cluster_labels


    @staticmethod
    def _e_step(X, centroids):
        distances_to_centroids = []
        for c in centroids:
            distances_to_centroids.append(np.sum((X - c) ** 2, axis=1))  # shape (M, )
        distances_to_centroids = np.array(distances_to_centroids) # shape (K, M)
        cluster_labels = np.argmin(distances_to_centroids, axis=0)  # shape (M, )
        inertia = np.sum(np.min(distances_to_centroids, axis=0))
        return cluster_labels, inertia

        
    def predict(self, X):
        assert self.was_fit      
        cluster_labels, _ = self._e_step(X, self.centroids)
        return cluster_labels
        


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans as skKMeans
    from sklearn.datasets import make_blobs

       
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(centers=3, n_samples=n_samples, random_state=random_state)

    # Correct number of clusters
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)

    plt.figure(figsize=(12, 12))
    
    # Incorrect number of clusters
    y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
    
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Incorrect Number of Blobs")
    
    # Anisotropicly distributed data
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
    
    plt.subplot(222)
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
    plt.title("Anisotropicly Distributed Blobs")
    
    # Different variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
    
    plt.subplot(223)
    plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
    plt.title("Unequal Variance")
    
    # Unevenly sized blobs
    X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_pred = KMeans(n_clusters=3,
                    random_state=random_state).fit_predict(X_filtered)
    
    plt.subplot(224)
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
    plt.title("Unevenly Sized Blobs")
    
    plt.show()
                    
                    
                    
                