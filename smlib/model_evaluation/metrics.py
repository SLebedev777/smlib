# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:51:54 2019

@author: NRA-LebedevSM
"""

import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)