# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:20:50 2019

@author: NRA-LebedevSM
"""

import numpy as np
import matplotlib.pyplot as plt

def bias_variance_regression(model, X_train, y_train, X_test, y_test, n_subsamples=10,
                  subsample_frac=.8, debug_plot=False):
    """
    Bias-variance decomposition of regression model EPE (expected prediction error)
    on every point of given test set.
    EPE(model, x0) = bias(model, x0)**2 + variance(model, x0) + random_data_noise
    """
    y_hat = np.zeros((n_subsamples, len(y_test)))
    residuals = np.zeros((n_subsamples, len(y_test)))
    subsample_size = int(subsample_frac * len(X_train))
    if debug_plot:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.scatter(X_test, y_test,  color='black')
    for k in range(n_subsamples):
        subsample = np.random.choice(len(X_train), subsample_size)
        Xs = X_train[subsample, :]
        ys = y_train[subsample]
        model.fit(Xs, ys)
        y_pred = model.predict(X_test)
        y_hat[k, :] = y_pred
        residuals[k, :] = y_test - y_pred
        if debug_plot: 
            plt.plot(X_test, y_pred)
    # calc bias and variance in every point of test set
    biases = np.mean(residuals, axis=0) ** 2
    variances = np.std(residuals, axis=0) ** 2
    if debug_plot:
        plt.subplot(3, 1, 2)
        plt.title('Biases**2 for points of test set')
        plt.scatter(X_test, biases)
        
        plt.subplot(3, 1, 3)
        plt.title('Variances for points of test set')
        plt.scatter(X_test, variances)
        plt.show()

    return biases, variances
