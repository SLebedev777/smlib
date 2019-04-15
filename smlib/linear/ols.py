# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:38:40 2019

@author: pups
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class LinearRegression:
    """
    Classic Ordinary Least Squares implementation
    """
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.fitted = False
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        if self.intercept:
            X = self.add_intercept(X)
        X_t = X.transpose()
        cov = np.matmul(X_t, X)
        self.coef_ = np.matmul(np.linalg.inv(cov), X_t).dot(y).transpose()
        self.rss_ = y - np.dot(X, self.coef_)
        self.fitted = True
        return self.coef_
    
    def predict(self, X_pred):
        X_pred = self.scaler.transform(X_pred)
        if self.intercept:
            X_pred = self.add_intercept(X_pred)
        return np.dot(X_pred, self.coef_)

    def regr_analysis(self):
        if not self.fitted:
            return
        print('Mean(RSS)=%.7f' % np.mean(self.rss_))
        s, pvalue = stats.normaltest(self.rss_)
        print('testing if RSS are normally distributed: p-value is %.5f' % pvalue)
        plt.scatter(range(len(self.rss_)), self.rss_)

    @staticmethod
    def add_intercept(X):
        f0 = np.ones((X.shape[0], 1))
        return np.hstack([f0, X])   
    
    @staticmethod
    def pinv(A):
        # matrix pseudo-inversion
        return np.linalg.pinv(A)


def bias_variance(model, X_train, y_train, X_test, y_test, n_subsamples=10,
                  subsample_frac=.8):
    """
    Bias-variance decomposition of model error on test set.
    Error(model, x0) = bias(model, x0)**2 + variance(model, x0) + data_noise
    """
    y_hat = np.zeros((n_subsamples, len(y_test)))
    residuals = np.zeros((n_subsamples, len(y_test)))
    subsample_size = int(subsample_frac * len(X_train))
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
        plt.plot(X_test, y_pred)
    # calc bias and variance in every point of test set
    biases = np.mean(residuals, axis=0) ** 2
    variances = np.var(y_hat, axis=0)

    plt.subplot(3, 1, 2)
    plt.title('Biases**2 for points of test set')
    plt.scatter(X_test, biases)
    
    plt.subplot(3, 1, 3)
    plt.title('Variances for points of test set')
    plt.scatter(X_test, variances)
    plt.show()
    
    return biases, variances
    
        
    
        
if __name__ == '__main__': 

    X = np.random.rand(100, 4)
    y = X[:, 2] * 2 + 4
    
    ols = LinearRegression()
    ols.fit(X, y)
    y_pred = ols.predict(X)
