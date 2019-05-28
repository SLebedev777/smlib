# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:26:57 2019

@author: pups
"""

import numpy as np
import matplotlib.pyplot as plt
from smlib.knn import kNN
from smlib.decision_trees.dt import DecisionTree
from smlib.model_evaluation.bias_variance import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=123)

complexity_param = range(1, 7)
models = [DecisionTree(max_depth=k) for k in complexity_param]

#test_errors, biases, variances = bias_variance_classification_fixed_model(models[0], X_train, y_train, 
#                                             X_test, y_test, n_subsamples=30)

EPE, B, V, Vu, Vb, EPE_check = bias_variance_classification(models, X_train, y_train, X_test, y_test, 
                                 n_subsamples=30)
