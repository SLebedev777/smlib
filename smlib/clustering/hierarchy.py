# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:45:06 2020

@author: Семен
"""
import numpy as np
import scipy
from scipy import cluster
from copy import deepcopy


        
class AgglomerativeClustering:
    def __init__(self, n_clusters=2, affinity='euclidean', linkage='complete'):
        self.n_clusters = n_clusters
        assert affinity in ['euclidean', 
                            #'l2', 'l1', 'cosine'
                            ]
        if affinity == 'euclidean':
            self.affinity = affinity
            self.affinity_func = lambda a, b: np.linalg.norm(a - b)

        assert linkage in ['complete', 
                           #'single', 'ward', 'average', 
                           ]
        self.linkage = linkage
 
    
    def fit(self, X):
        def lf_complete(ci, cj):
            D = np.zeros((len(ci), len(cj)))
            for row, i in enumerate(ci):
                for col, j in enumerate(cj):
                    D[row, col] = self.affinity_func(X[i], X[j])
            return D.max()
            
        if self.linkage == 'complete':
            self.linkage_func = lf_complete
                    
        M = len(X)
        CLINFO = [{i: [i] for i in range(M)}]  # clusters info for iterations
        L = -np.ones((M, M))
        # fill upper triangle matrix of linkages between points
        cc = CLINFO[0]
        for i in range(M):
            for j in range(M):
                if i >= j: 
                    continue
                L[i, j] = self.linkage_func(cc[i], cc[j])
        L_max = L.max()
        L[L < 0] = L_max
        
        Z = np.zeros((M-1, 4))  # linkage matrix in scipy format
                
        # if we want to build dendrogram using scipy, we must build special
        # linkage matrix in scipy format.
        # in our implementation of agglomerative clustering, we merge 2 best clusters
        # at iteration t, and place union result into 1st cluster, and delete 2nd.
        # scipy doesn't remove any clusters, but creates new ones by merging.
        # so, at every iteration t we must record new index of won cluster as:
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
                if cid < ibest:
                    L[cid, ibest] = l
                elif cid > ibest:
                    L[ibest, cid] = l
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

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X = iris.data
    
    model = AgglomerativeClustering()
    model.fit(X)
    
    plt.figure()
    cluster.hierarchy.dendrogram(model.Z_, truncate_mode='level', p=3)
    plt.show()