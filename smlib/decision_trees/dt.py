# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 00:10:53 2019

@author: LebedevSM
"""

import logging
from scipy.stats import entropy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

class TreeNode:
    def __init__(self, key, depth, val=None, left=None, right=None, parent=None):
        self.key = key
        self.depth = depth
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent
        self.id = str(np.random.randint(65535))
    
    def __str__(self):
        str_left   = 'None' if self.left is None else self.left.id
        str_right  = 'None' if self.right is None else self.right.id
        str_parent = 'None' if self.parent is None else self.parent.id
        return 'node id=%s  key=%s  depth=%d  val=%s  left.id=%s  right.id=%s  parent.id=%s' % \
                (self.id, self.key, self.depth, self.val, str_left, str_right, str_parent)
    
   
def categorical_entropy(y):
    # y is series
    probas = y.value_counts() / len(y)
    return entropy(probas.values)

def gini(y):
    # y is series
    probas = y.value_counts() / len(y)
    return 1 - np.sum((probas**2).values)

def mse(y):
    # y is series
    y_mean = np.mean(y.values)
    err = (y - y_mean)**2
    return np.mean(err.values)


class DecisionTree:
    def __init__(self, task='classification', criterion='entropy', 
                 max_depth=3, min_samples_leaf=5, verbose=False):
        self.task = task
        self.criterion = criterion
        if self.task == 'classification':
            if self.criterion == 'entropy':
                self.metric_func = categorical_entropy
            elif self.criterion == 'gini':
                self.metric_func = gini
            else:
                raise ValueError ('wrong criterion %s for task %s' % (criterion, task))
            self.label_func = lambda y: y.value_counts().index[0]
        elif self.task == 'regression':
            if self.criterion == 'mse':
                self.metric_func = mse
                self.label_func = lambda y: np.mean(y)
            else:
                raise ValueError ('wrong criterion %s for task %s' % (criterion, task))
        else:
            raise ValueError ('wrong task %s' % task)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.current_node = None
        self.root = None
        self.nodes = []
        self.size = 0
        self.verbose = verbose
        
    def fit(self, X, y):
        # for fit(), X and y are specified in sklearn fashion:
        # X is numpy array of shape (n_samples, n_features)
        # y is numpy array of shape (n_samples)
        # then they are transformed to pandas objects and passed to _build()
        assert len(X) == len(y)
        dfX = pd.DataFrame(X)
        dfy = pd.Series(y)
        self._build(dfX, dfy, current_node=None)
    
    def _predict_one(self, x):
        # x is pd.Series with 1 row or numpy 1d array with shape (n_features,)
        if self.root is None:
            raise ValueError ('trying to predict on empty tree')
        current_node = self.root
        while True:
            if current_node.key == 'leaf':
                break
            else:
                feature, threshold, inform_gain = current_node.key
                if x[feature] <= threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
        return current_node.val

    def predict(self, x):
        # x is pd.DataFrame with many rows or 
        # numpy 2d array with shape (n_samples, n_features)
        x_iter = x.iterrows() if isinstance(x, pd.DataFrame) else enumerate(x)
        return np.array([self._predict_one(row) for _, row in x_iter])


    def find_best_current_split(self, x, y, features):
        N = len(y)
        calcs = []
        for feature in features:
            feature_values = sorted(x[feature].unique().tolist())
            thresholds = feature_values[:-1]
            if not thresholds:
                calcs.append([feature, feature_values[0], 0])
            for threshold in thresholds:                
                mask = x[feature] <= threshold
                left_y, right_y = y[mask], y[~mask]
                y_metric = self.metric_func(y)
                left_metric = self.metric_func(left_y)
                right_metric = self.metric_func(right_y)
                ig = y_metric - (len(left_y)*left_metric + len(right_y)*right_metric) / N
                logger.debug('feature: %s  threshold: %.2f  y_metric = %.3f  left_y_metric = %.3f  right_y_metric = %.3f  ig = %.3f' %
                              (feature, threshold, y_metric, left_metric, right_metric, ig))
                calcs.append([feature, threshold, ig])
        key = sorted(calcs, key=lambda x: x[2], reverse=True)[0]
        best_feature = key[0]
        best_threshold = key[1]
        best_ig = key[2]
        if best_ig == 0.0:
            return key, x, y, x, y
        mask = x[best_feature] <= best_threshold
        left_x, left_y = x[mask], y[mask]
        right_x, right_y = x[~mask], y[~mask]
        logger.debug('best split = %s' % key)
        logger.debug('%s' % left_y.value_counts())
        logger.debug('%s' % right_y.value_counts())
        return key, left_x, left_y, right_x, right_y
    
    def create_node(self, key, val, current_node):
        if current_node is None:
            # create root
            node = TreeNode(key=key, depth=1, val=val)
            self.root = node
            self.current_node = node
        else:
            self.current_node = current_node
            depth = current_node.depth + 1
            node = TreeNode(key, depth, val, parent=self.current_node)
            if self.current_node.left is None and self.current_node.right is None:
                self.current_node.left = node
            elif self.current_node.left is not None and self.current_node.right is None:
                self.current_node.right = node
        self.size += 1
        self.nodes.append(node)
        return node

    def _build(self, x, y, current_node):
        # x is dataframe, y is series
        logger.debug('current_node: %s' % current_node)
        y_metric = self.metric_func(y)
        curr_depth = current_node.depth if current_node is not None else 0
        if curr_depth >= self.max_depth or len(y) <= max(1, self.min_samples_leaf) or y_metric < 0.001:
            # make leaf with class label
            label = self.label_func(y)
            node = self.create_node('leaf', label, current_node)
            logger.debug('creating leaf!')
            logger.debug('current depth = %d  max_depth = %d' % (curr_depth, self.max_depth))
            logger.debug('len_y = %d  y metric = %.3f class label = %s' % (len(y), y_metric, label))
        else:
            logger.debug('finding split for current node:  %s' % str(current_node))
            features = x.columns
            key, left_x, left_y, right_x, right_y = self.find_best_current_split(x, y, features)
            node = self.create_node(key, None, current_node)
            self._build(left_x, left_y, node)
            self._build(right_x, right_y, node)
        return node
