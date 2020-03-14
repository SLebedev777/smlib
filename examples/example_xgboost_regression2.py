# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 01:05:32 2020

@author: Семен
"""
import numpy as np
from smlib.boosting.xgb_regr import XGBoostRegressor
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor as GDBSklearn
from sklearn.metrics.regression import mean_squared_error as mse

data = datasets.fetch_california_housing()
X = np.array(data.data)[:10000]
y = np.array(data.target)[:10000]
print(X.shape)

skgb = GDBSklearn(max_depth=3,n_estimators=150,learning_rate=0.2)
xgb = XGBoostRegressor(max_depth=3,n_estimators=20, tree_method='hist') 

from sklearn.model_selection import KFold

def get_metrics(X,y,n_folds=2, model=None):

    kf = KFold(n_splits=n_folds, shuffle=True)
    kf.get_n_splits(X)

    er_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train,y_train)
        predict = model.predict(X_test)
        er_list.append(mse(y_test, predict))
    
    return er_list

for name, clf in [('Smlib XGBoost', xgb),
                  ('Sklearn GB', skgb)
                  ]:
    print(name, np.mean(get_metrics(X, y, 5, clf)))