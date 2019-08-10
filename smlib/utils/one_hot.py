#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:51:20 2019

@author: sl
"""
import numpy as np

class OneHotEncoder:
    """
    Transform array of values of any type to binary dummy matrix.
    """
    def __init__(self):
        self.unique = None
    
    def fit(self, vector):
        assert len(vector.shape) == 1
        
        self.unique = np.unique(vector)
 
    def transform(self, vector):
        assert len(vector.shape) == 1
        assert self.unique is not None
        assert all(v in self.unique for v in np.unique(vector))
        
        one_hot = np.zeros((len(vector), len(self.unique)))
        for i, v in enumerate(vector):
            one_hot[i, np.where(self.unique == v)] = 1.
        
        return one_hot
    
    def fit_transform(self, vector):
        assert len(vector.shape) == 1
        
        self.unique = np.unique(vector)        
        one_hot = np.zeros((len(vector), len(self.unique)))
        for i, v in enumerate(vector):
            one_hot[i, np.where(self.unique == v)] = 1.
        
        return one_hot
    
    def inverse_transform(self, one_hot):
        assert self.unique is not None
        assert one_hot.shape[1] == len(self.unique)
        
        return np.array([self.unique[np.where(row == 1)] 
                                     for row in one_hot]).reshape(len(one_hot))

    
def test_one_hot():
    a = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    b = np.array([2, 0, 1, 1])
    c = np.array(['a', 'b', 'c', 'a', 'b'])
    d = np.array(['b', 'b', 'a'])
    
    oh = OneHotEncoder()
    a1 = oh.fit_transform(a)
    print(a1)
    print(oh.inverse_transform(a1) == a)
    print(oh.transform(b))
    
    print('-'*50)
    
    oh2 = OneHotEncoder()
    oh2.fit(c)
    c1 = oh2.transform(c)
    print(c1)
    print(oh2.transform(d))
    print(oh2.inverse_transform(c1) == c)
    
