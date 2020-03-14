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

from smlib.decision_trees.dt import TreeNode, DecisionTree

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


class XGBRegressionTree(DecisionTree):
    """
    Regression Tree specially for XGBoost.
    Should not be used for stand-alone usage.
    Split criteria is "Gain" (see XGBoost theory).
    Loss function is MSE.
    All formulas (node objective, leaf value...) are explicitly calculated for MSE.
    Simple histogram-based split is supported.
    """
    def __init__(self, gamma=0.1,  lambd=1.0, max_depth=3, tree_method='bruteforce'):
        self.max_depth = max_depth
        self.current_node = None
        self.root = None
        self.nodes = []
        self.size = 0
        self.gamma = gamma
        self.lambd = lambd
        assert tree_method in ['bruteforce', 'hist']
        self.tree_method = tree_method
        
    def _node_objective(self, y):
        """
        Calculate part of objective G**2 / (H + lambd) in a tree node for MSE loss.
        """
        G = np.sum(y)
        H = len(y)
        return G ** 2 / (H + self.lambd)

    def _leaf_weight(self, y):
        """
        Calculate optimal leaf value for MSE loss.
        """
        G = np.sum(y)
        H = len(y)
        return -G / (H + self.lambd)        

    def _get_thresholds_for_split(self, x, y, feature):
        thresholds = []
        if self.tree_method == 'bruteforce':
            feature_values = sorted(x[feature].unique().tolist())
            thresholds = feature_values[:-1]
        elif self.tree_method == 'hist':
            hist_bin_edges = np.histogram_bin_edges(x[feature], bins='sqrt')
            thresholds = hist_bin_edges.tolist()[:-1]
        return thresholds
          

    def _find_best_current_split(self, x, y, features):
        calcs = []
        node_obj = self._node_objective(y)
        for feature in features:
            thresholds = self._get_thresholds_for_split(x, y, feature)
            if not thresholds:
                calcs.append([feature, np.min(x[feature]), 0])
            for threshold in thresholds:                
                mask = x[feature] <= threshold
                left_y, right_y = y[mask], y[~mask]
                left_obj = self._node_objective(left_y)
                right_obj = self._node_objective(right_y)
                gain = 0.5*(left_obj + right_obj - node_obj) - self.gamma
                if gain < 0.0:
                    gain = 0.0
                calcs.append([feature, threshold, gain])
        if not calcs:
            # could not do split - return to make a leaf here
            return [0, 0, 0], x, y, x, y
        key = sorted(calcs, key=lambda x: x[2], reverse=True)[0]
        best_feature = key[0]
        best_threshold = key[1]
        best_gain = key[2]
        if best_gain == 0.0:
            return key, x, y, x, y
        mask = x[best_feature] <= best_threshold
        left_x, left_y = x[mask], y[mask]
        right_x, right_y = x[~mask], y[~mask]
        return key, left_x, left_y, right_x, right_y
    

    def _build(self, x, y, current_node):
        # x is dataframe, y is series
        node_obj = self._node_objective(y)
        curr_depth = current_node.depth if current_node is not None else 0
        if curr_depth >= self.max_depth or \
        (current_node is not None and current_node.key == [0, 0, 0]) or node_obj < 1e-9:
            # make leaf
            w = self._leaf_weight(y)
            node = self.create_node('leaf', w, current_node)
        else:
            features = x.columns
            key, left_x, left_y, right_x, right_y = self._find_best_current_split(x, y, features)
            node = self.create_node(key, None, current_node)
            # update feature importance
            best_feature, best_threshold, best_gain = key
            self.feature_importances_[best_feature] += best_gain*len(y)

            self._build(left_x, left_y, node)
            self._build(right_x, right_y, node)
        return node
