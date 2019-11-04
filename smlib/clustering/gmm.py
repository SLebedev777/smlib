# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:26:56 2019

@author: Семен
"""
import numpy as np
from .kmeans import KMeans

class GaussianMixtureEM:
    def __init__(self, n_components, eps=1e-4, n_init=5, max_iter=300, random_state=None,
                 cov_mode='diag', init_mode='kmeans'):
        self.n_components = n_components
        self.eps = eps
        self.n_init = n_init
        self.max_iter = max_iter
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif type(random_state) == int:
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        assert cov_mode == 'diag', 'Only diagonal covariance matrices are supported'
        self.cov_mode = cov_mode
        self.init_mode = init_mode
        self.was_fit = False

   
    def fit(self, X):
        M, N = X.shape
        K = self.n_components
        best_llh = 0

        for attempt in range(self.n_init):
            # init parameters of mixture components
            if self.init_mode == 'random':
                mu = X[self.random_state.choice(range(M), size=K), :]  # shape = (K, N)
            elif self.init_mode == 'kmeans':
                kmeans = KMeans(n_clusters=K, n_init=self.n_init, random_state=self.random_state)
                kmeans.fit(X)
                mu = kmeans.centroids  # shape = (K, N)
            w = np.array([1 / K] * K)
            cov = [np.eye(N) for _ in range(K)]  # init array of covariances for each component
            
            prev_llh = 0
            for it in range(self.max_iter):
                # E step
                p = []
                for k in range(K):
                    cov_inv = np.linalg.inv(cov[k])
                    cov_det = np.linalg.det(cov[k])
                    p.append(w[k] * gaussian_pdf(X, mu[k], cov_inv, cov_det, N))
                p = np.array(p)  # shape = (K, M)
                gm_probas = np.sum(p, axis=0)  # p(x). shape = (M,)
                resp = np.divide(p, gm_probas)  # responsibilities. shape = (K, M)
                
                llh = np.sum(np.log(gm_probas))
                if prev_llh and (llh - prev_llh <= self.eps):
                    break
                prev_llh = llh
                
                # M step
                sum_resp = np.sum(resp, axis=1)  # shape = (K, )
                mu = np.divide(resp.dot(X).T, sum_resp).T  # shape = (K, N)
                w = sum_resp / float(M)  # shape = (K, )
                if self.cov_mode == 'diag':
                    for k in range(K):
                        Xc2 = (X - mu[k]) ** 2
                        diag_covs = np.divide(resp[k].dot(Xc2).T, sum_resp[k]).T  # (N,)
                        cov[k] = np.diag(diag_covs)
            
            if attempt == 0 or (best_llh and llh > best_llh):
                best_model = [resp, mu, cov, w]
                best_llh = llh

        self.resp_, self.mu_, self.cov_, self.w_ = best_model        
        self.cluster_labels_ = np.argmax(self.resp_, axis=0)        
        self.was_fit = True

    def fit_predict(self, X):
        self.fit(X)
        return self.cluster_labels_
        
    def predict(self, X):
        print('Not implemented')
        return None


def gaussian_pdf(X, mu, cov_inv, cov_det, n):
    X_mu = X - mu
    md_sqr = ((X_mu @ cov_inv) * X_mu).sum(axis=1)  # squared Mahalanobis distances
    p = np.exp(-0.5 * md_sqr) / np.sqrt(cov_det * (2*np.pi)**n)
    return p
