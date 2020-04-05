# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:45:06 2020

@author: Семен
"""
import numpy as np
from numpy.linalg import norm as npnorm
from copy import deepcopy


        
class AgglomerativeClustering:
    """
    Bottom-up Hierarchical Clustering.
    """
    def __init__(self, affinity='euclidean', linkage='complete',
                 cut_mode='n_clusters', cut_value=3):
        assert affinity in ['euclidean', 
                            'l2', 'l1', 'cosine'
                            ]
        self.affinity = affinity
        if affinity == 'euclidean':
            self.affinity_func = lambda a, b: npnorm(a - b)
        elif affinity == 'l2':
            self.affinity_func = lambda a, b: npnorm(a - b) ** 2
        elif affinity == 'l1':
            self.affinity_func = lambda a, b: npnorm(a - b, ord=1)
        elif affinity == 'cosine':
            self.affinity_func = lambda a, b: 1. - (np.dot(a, b) / (npnorm(a) * npnorm(b)))

        assert linkage in ['complete', 'single', 'average', 
                           #'ward' 
                           ]
        self.linkage = linkage
        
        assert cut_mode in ['n_clusters',
                            'dist_thres'
                            ]
        self.cut_mode = cut_mode
        
        assert cut_value > 0
        self.cut_value = cut_value
 
    
    def fit(self, X):
        
        def _lf(ci, cj, mode):
            """
            Linkage function that measures distance between 2 clusters.
            """
            ni = len(ci)
            nj = len(cj)
            D = np.zeros((ni, nj))
            for row, i in enumerate(ci):
                for col, j in enumerate(cj):
                    D[row, col] = self.affinity_func(X[i], X[j])
            if mode == 'complete':
                return D.max()
            if mode == 'single':
                return D.min()
            if mode == 'average':
                return D.sum() / (ni * nj)

            
        if self.linkage in ['single', 'complete', 'average']:
            self.linkage_func = lambda ci, cj: _lf(ci, cj, self.linkage)
                    
        M = len(X)
        CLINFO = [{i: [i] for i in range(M)}]  # clusters info for iterations
        L = -np.ones((M, M))
        # fill upper triangle matrix of linkages between singleton data points
        cc = CLINFO[0]
        for i in range(M):
            for j in range(M):
                if i >= j: 
                    continue
                L[i, j] = self.linkage_func(cc[i], cc[j])
        L_max = L.max()
        L[L < 0] = L_max
        
        Z = np.zeros((M-1, 4))  # create empty linkage matrix in scipy format
                
        # if we want to build dendrogram using scipy, we must build special
        # linkage matrix in scipy format (for details, see scipy.cluster.hierarchy).
        # in our implementation of agglomerative clustering, we merge 2 best clusters
        # at iteration t, and place union result into 1st cluster, and delete 2nd.
        # scipy doesn't remove any clusters, but creates new ones by merging.
        # so, at every iteration t, newly formed cluster gets its new index as:
        #    i (in fact i' after merging)  ->  M + t - 1
        new_old = {i: i for i in range(M)}
        
        for t in range(1, M-1):
            # find pair of nearest clusters
            best_tuple = np.unravel_index(np.argmin(L), L.shape)
            Lbest = np.min(L)
            ibest = min(best_tuple)
            jbest = max(best_tuple)
            # always merge cluster with maximum index TO a cluster with min index
            cc = deepcopy(CLINFO[-1])  # current dict of clusters
            cc[ibest] = cc[ibest] + cc[jbest]  # merge clusters into 1st cluster
            cc.pop(jbest)  # delete 2nd cluster
            CLINFO.append(cc)
            # recalculate linkages between new cluster and other existing clusters
            for cid, c in cc.items():
                l = self.linkage_func(cc[ibest], c)
                # fill column and row in triangle linkage matrix
                if cid < ibest:
                    L[cid, ibest] = l
                elif cid > ibest:
                    L[ibest, cid] = l
            # exclude j-th cluster from calculations
            L[jbest, :] = L_max
            L[:, jbest] = L_max
            
            # fill row of scipy linkage matrix to save clustering info
            i_new, j_new = new_old[ibest], new_old[jbest]
            Z[t-1, 0] = min(i_new, j_new)
            Z[t-1, 1] = max(i_new, j_new)
            Z[t-1, 2] = Lbest  # distance between best clusters
            Z[t-1, 3] = len(cc[ibest])  # number of observations in new cluster
            new_old[ibest] = M + t - 1
        
        # fill last row of linkage matrix with info about 2 left clusters
        i_new, j_new = [new_old[k] for k in CLINFO[-1].keys()]
        Z[M-2, 0] = min(i_new, j_new)
        Z[M-2, 1] = max(i_new, j_new)
        Z[M-2, 2] = self.linkage_func(*CLINFO[-1].values())
        Z[M-2, 3] = M
        
        self.Z_ = Z

        # assign cluster labels to data points, according to cut parameters
        labels = np.zeros(M, dtype=int)
        if self.cut_mode == 'dist_thres':
            # we imaginally cut dendrogram at distance level 'cut_value',
            # and number of intersections gives us number of clusters
            for t in range(M-2, 0, -1):
                if Z[t, 2] <= self.cut_value:
                    cc = CLINFO[min([t+1, M-2])]
                    for i, (cid, c) in enumerate(cc.items()):
                        labels[c] = i                    
                    break
        if self.cut_mode == 'n_clusters':
            # we select appropriate number of clusters at dendrogram, 
            # going down from root
            assert 1 <= self.cut_value <= M
            for t in range(M-2, 0, -1):
                cc = CLINFO[t]
                if len(cc) == self.cut_value:
                    for i, (cid, c) in enumerate(cc.items()):
                        labels[c] = i
                    break
        if not labels.any():
            raise ValueError("Could not assign labels with specified cut parameters")
        self.labels_ = labels
