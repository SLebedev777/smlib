# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:06:07 2019

@author: pups
"""

import numpy as np
import logging

def svd(X):
    U, S, VT = np.linalg.svd(X)
    return U, S, VT
