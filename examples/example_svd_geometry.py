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
v1 = v.dot(X)

u, s, vt = svd(X)
vtv = vt.dot(v.T)
svtv = np.diag(s).dot(vtv)
usvtv = u.dot(svtv).reshape(v.shape)

origin = [0], [0]
plt.quiver(*origin, v[:,0], v[:,1], color=['k'], scale_units='y', scale=50)
plt.legend('source vectors: v')
plt.quiver(*origin, v1[:,0], v1[:,1], color=['g'], scale_units='y', scale=50)
plt.legend('after operator: X*v')
plt.quiver(*origin, vtv[0, :], vtv[1, :], color=['r'], scale_units='y', scale=50)
plt.legend('1st rotation: Vt*v')
plt.quiver(*origin, svtv[0, :], svtv[1, :], color=['b'], scale_units='y', scale=50)
plt.legend('scaling: S*Vt*v')
plt.quiver(*origin, usvtv[:, 0], usvtv[:, 1], color=['y'], scale_units='y', scale=50)
plt.legend('2nd rotation: U*S*Vt*v')
