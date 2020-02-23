# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:26:37 2020

@author: Семен
"""
import numpy as np
from smlib.boosting.gb_regr import GBRegressor
import matplotlib.pyplot as plt

X = np.linspace(-10, 10)
y = np.sin(X) + 3
X = X[:, np.newaxis]

plt.figure(figsize=(10, 7))
plt.scatter(X, y)

for n in [2, 5, 10, 50, 500]:
    gbr = GBRegressor(n, max_depth=3)
    gbr.fit(X, y)
    y_pred = gbr.predict(X)
    plt.plot(X, y_pred, label=n)
    plt.legend()
