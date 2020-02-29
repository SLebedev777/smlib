# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:56:31 2019

@author: NRA-LebedevSM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smlib.decision_trees.dt import DecisionTree
from smlib.bagging.random_forest import RandomForest
from sklearn.ensemble import RandomForestRegressor as skRFR

# Create a random dataset
rng = np.random.RandomState(205)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


# Fit regression model
#regr_1 = DecisionTree(task='regression', criterion='mse', max_depth=1)
dt = DecisionTree(task='regression', criterion='mse', max_depth=15, min_samples_leaf=3,
                      verbose=True)
rf_params = {'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf': 5}
rf = RandomForest(task='regression', **rf_params)
skrf = skRFR(**rf_params)
dt.fit(X, y)
rf.fit(X, y)
skrf.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_dt = dt.predict(X_test)
y_rf = rf.predict(X_test)
y_skrf = skrf.predict(X_test)

# Plot the results
plt.figure(1, (15, 10))
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_dt, color="cornflowerblue",
         label="DecisionTree", linewidth=2)
plt.plot(X_test, y_rf, color="r", label="RandomForest", linewidth=2)
plt.plot(X_test, y_skrf, color="yellowgreen", label="sklearn RandomForest", linewidth=2)

plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression VS RandomForest")
plt.legend()
plt.show()