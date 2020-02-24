# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:56:31 2019

@author: NRA-LebedevSM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smlib.decision_trees.dt import DecisionTree

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


# Fit regression model
regr_1 = DecisionTree(task='regression', criterion='mse', max_depth=1)
regr_2 = DecisionTree(task='regression', criterion='mse', max_depth=15, min_samples_leaf=1,
                      verbose=True)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
dfX_test = pd.DataFrame(X_test, columns=['X'])
y_1 = np.array(regr_1.predict(dfX_test))
y_2 = np.array(regr_2.predict(dfX_test))

# Plot the results
plt.figure(1, (15, 10))
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=15", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()