# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:19:29 2019

@author: pups
"""
import numpy as np
import matplotlib.pyplot as plt
from smlib.core.linalg import svd

X = np.array([[ 0.96,  1.72],
              [ 2.28,  0.96]])
'''
X = np.array([[ 3,  0],
              [ 0,  2]])
'''
v = np.array([[1, 0], [0, 1], [1, 1]])
v1 = np.matmul(X, v.T)

u, s, vt = svd(X)
#u, s, vt = np.linalg.svd(X)
vtv = np.matmul(vt, v.T)
svtv = np.matmul(np.diag(s), vtv)
usvtv = np.matmul(u, svtv)

origin = [0], [0]
f, axarr = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(8, 8))
f.suptitle('SVD geometry\nX * v = U * S * Vt * v')
axarr[0, 0].set_title('source vectors v and operator apply X*v')
axarr[0, 0].quiver(*origin, v[:,0], v[:,1], color=['k'], scale_units='y', scale=60)
axarr[0, 0].quiver(*origin, v1[0, :], v1[1, :], color=['g'], scale_units='y', scale=60)
axarr[1, 0].quiver(*origin, v[:,0], v[:,1], color=['k'], scale_units='y', scale=60)
axarr[1, 0].quiver(*origin, v1[0, :], v1[1, :], color=['g'], scale_units='y', scale=60)
axarr[2, 0].quiver(*origin, v[:,0], v[:,1], color=['k'], scale_units='y', scale=60)
axarr[2, 0].quiver(*origin, v1[0, :], v1[1, :], color=['g'], scale_units='y', scale=60)
axarr[0, 1].quiver(*origin, vtv[0, :], vtv[1, :], color=['r'], scale_units='y', scale=60)
axarr[0, 1].set_title('1st rotation: Vt*v')
axarr[1, 1].quiver(*origin, svtv[0, :], svtv[1, :], color=['b'], scale_units='y', scale=60)
axarr[1, 1].set_title('scaling: S*Vt*v')
axarr[2, 1].quiver(*origin, usvtv[0, :], usvtv[1, :], color=['g'], scale_units='y', scale=60)
axarr[2, 1].set_title('2nd rotation: U*S*Vt*v')
