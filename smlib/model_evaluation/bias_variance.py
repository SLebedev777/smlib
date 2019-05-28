# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:20:50 2019

@author: NRA-LebedevSM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bias_variance_regression_fixed_model(model, X_train, y_train, X_test, y_test, n_subsamples=30,
                             noise_ratio=0.1, debug_plot=False):
    """
    Bias-variance decomposition of regression model EPE (expected prediction error)
    on every point of given test set.
    EPE(model, x0) = bias(model, x0)**2 + variance(model, x0) + random_data_noise
    For model with fixed complexity C:
        1. make bootstrap samples of train set
        2. train model on every bootstrap sample
        3. calculate bias and variance in every point of test set
    """
    y_hat = np.zeros((n_subsamples, len(y_test)))
    residuals = np.zeros((n_subsamples, len(y_test)))
    M = len(X_train)
    all_train_indices = set(range(M))
    for k in range(n_subsamples):
        bootstrap_indices = np.random.choice(M, M, replace=True)
        Xs = X_train[bootstrap_indices, :]
        ys = y_train[bootstrap_indices]
        out_of_bag_indices = sorted(all_train_indices - set(bootstrap_indices))
        X_oob = X_train[out_of_bag_indices]
        y_oob = y_train[out_of_bag_indices]
        model.fit(Xs, ys)
        y_pred = model.predict(X_test)
        y_hat[k, :] = y_pred
        residuals[k, :] = y_pred - y_test
    # calc bias and variance in every point of test set
    test_errors = np.mean(residuals ** 2, axis=0)
    biases = np.mean(residuals, axis=0) ** 2
    variances = np.std(residuals, axis=0) ** 2
    if debug_plot:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.scatter(X_test, test_errors,  color='r')
        plt.subplot(3, 1, 2)
        plt.title('Biases**2 for points of test set')
        plt.scatter(X_test, biases, color='b')
        plt.subplot(3, 1, 3)
        plt.title('Variances for points of test set')
        plt.scatter(X_test, variances, color='g')
        plt.show()

    return test_errors, biases, variances 

def bias_variance_regression(models, X_train, y_train, X_test, y_test, n_subsamples=30,
                             noise_ratio=.0):
    """
    For each model in list of models with varying complexity C:
        1. calculate biases and variances in every point of test set
        2. calculate average bias and variance over whole test set
    Finally, for this class of models we get functions:
        B(C) = average_bias(C, train_set, test_set)
        V(C) = average_variance(C, train_set, test_set)
    for this given data set.

    """
    EPE = []
    B = []
    V = []

    for model in models:
        test_errors, biases, variances = bias_variance_regression_fixed_model(
                model, X_train, y_train, X_test, y_test, n_subsamples,
                             noise_ratio, debug_plot=False)
        EPE.append(np.mean(test_errors))
        B.append(np.mean(biases))
        V.append(np.mean(variances))

    return EPE, B, V


def bias_variance_classification_fixed_model(model, X_train, y_train, 
                                             X_test, y_test, n_subsamples=30):
    """
    Bias-variance decomposition of classification model EPE (expected prediction error)
    on every point of given test set (according to Pedro Domingos theory)
    For model with fixed complexity C:
        1. make n_subsamples bootstrap samples of train set
        2. train model on every bootstrap sample
        3. calculate bias and variance in every point of test set
    """
    loss_func_point  = lambda y1, y2: 0 if y1 == y2 else 1  # 0-1 loss
    loss_func_vector = lambda y1, y2: np.array([loss_func_point(y1[j], y2[j]) for j in range(len(y1))])
    
    y_hat = np.zeros((n_subsamples, len(y_test)))
    losses = np.zeros((n_subsamples, len(y_test)))
    M = len(X_train)
    all_train_indices = set(range(M))
    for k in range(n_subsamples):
        print(k)
        bootstrap_indices = np.random.choice(M, M, replace=True)
        Xs = X_train[bootstrap_indices]
        ys = y_train[bootstrap_indices]
        out_of_bag_indices = sorted(all_train_indices - set(bootstrap_indices))
        X_oob = X_train[out_of_bag_indices]
        y_oob = y_train[out_of_bag_indices]
        model.fit(Xs, ys)
        y_pred = model.predict(X_test)
        y_hat[k, :] = y_pred
        losses[k, :] = loss_func_vector(y_test, y_pred)
    # calc bias and variance in every point of test set for 0-1 loss
    test_errors = np.mean(losses, axis=0)
    df_y_hat = pd.DataFrame(y_hat)
    # for 0-1 loss, main prediction in point x is most common predicted label in this point
    main_predictions = df_y_hat.apply(lambda point: point.value_counts().index[0], axis=0).values
    # bias in point x is loss between true label and main prediction in this point
    biases = loss_func_vector(y_test, main_predictions)
    # variance in point x is mean loss between every prediction and main prediction
    # Var(x) = E(Loss01(y_pred, main_prediction)) = P(y_pred != main_prediction)
    variances = np.array([np.mean([loss_func_point(yh, main_predictions[i]) for yh in df_y_hat.iloc[:, i].values])
                for i in range(len(X_test))])

    return test_errors, biases, variances 

def bias_variance_classification(models, X_train, y_train, X_test, y_test, 
                                 n_subsamples=30):
    """
    Only 0-1 loss is supported.
    Noise is ignored.
    For each model in list of models with varying complexity C:
        1. calculate biases and variances in every point of test set
        2. calculate avg bias, avg unbiased variance, and 
        avg biased variance over whole test set
    Finally, for this class of models we get functions:
        B(C) = average_bias(C)
        V(C) = average_unbiased_variance(C) - average_biased_variance(C)
    for this given data set.

    """
    EPE = []
    B = []
    V = []
    Vu = []
    Vb = []
    EPE_check = []

    N = len(X_test)
    for model in models:
        print(model)
        test_errors, biases, variances = bias_variance_classification_fixed_model(
                model, X_train, y_train, X_test, y_test, n_subsamples)
        EPE.append(np.mean(test_errors))
        B.append(np.mean(biases))  # = Nb/N (see lower)
        biases = biases.astype(int)
        unbiased_points = np.where(biases == 0)[0]
        biased_points = np.where(biases > 0)[0]
        Nu = unbiased_points.size
        Nb = biased_points.size
        vu = np.sum(variances[unbiased_points])/N if Nu > 0 else 0  # avg unbiased variance
        vb = np.sum(variances[biased_points])/N if Nb > 0 else 0  # avg biased variance
        V.append(vu - vb)  # Vu increases error, while Vb decreases error!
        Vu.append(vu)
        Vb.append(vb)
        EPE_check.append(vu - vb + Nb/N)

    return EPE, B, V, Vu, Vb, EPE_check
