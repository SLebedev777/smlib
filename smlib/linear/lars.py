# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:45:30 2019

@author: Семен
"""
import numpy as np
import matplotlib.pyplot as plt
from smlib.utils.standardize import Standardizer
from sklearn.preprocessing import StandardScaler

class LARS:
    """
    Least Angle Regression (by Efron, Hastie, Tibshirani, 2002)
    """
    def __init__(self, method='lars', max_steps=500):
        self.standardizer_X = Standardizer()
        self.standardizer_y = Standardizer(with_std=False)
        self.method = method
        self.max_steps = max_steps
    
    def fit(self, X, y):
        M, N = X.shape
        assert len(y) == M
        X = self.standardizer_X.fit_transform(X)
        y = self.standardizer_y.fit_transform(y.reshape(-1, 1)).reshape((M,))
        
        self.alphas_ = []
        w = np.zeros((N,))
        mu = np.zeros((M,))  # mu = np.dot(w, X) fit vector for active set
        A = []
        coef_path = [w.tolist()]
        eps = 1e-20
        step = 0
        add = True
        while step < self.max_steps:
            if len(A) >= N:
                break
            r = y - mu  # residual after previous step
            if np.linalg.norm(r) <= eps:
                break
            if add:
                c = np.dot(X.T, r)  # vector of covariances between r and each feature
                abs_c = np.abs([cc if ind not in A else 0.0 for ind, cc in enumerate(c)])
                C = np.max(abs_c)  # value of absolute maximum covariance with residual,
                # for un-active variables.
                # For current step, C = lambda, in terms of L1-regularization term in loss:
                # lambda * || w || L1
                # As we build weights path from total zeros to full OLS solution,
                # lambda decreases from +Inf towards 0, as covariances do.
                d = np.where(abs_c == C)[0][0]  # index(es) of features with max covariance with residual
                A.append(d)
                print(f'Step: {step}  Added: {d}  C: {C}')
            elif not add and self.method == 'lasso':
                # in Lasso, if some weight intersected zero and we removed it
                # from the active set, we must make a step without adding
                # new variable.
                c = np.dot(X.T, r)
                abs_c = np.abs(c[A])
                C = np.max(abs_c)
                print(f'Step: {step}  Added: None, active set frozen  C: {C}')
            s = np.sign(c[A])
            XA = np.multiply(X[:, A], s.T)
            G = np.dot(XA.T, XA)  # covariance between current best features
            oA = np.ones((len(A)))
            G_1 = np.linalg.inv(G)
            AA = (oA.T @ G_1 @ oA)**(-0.5)  # inverse in OLS solution
            wA = AA * G_1 @ oA.T  # get OLS solution for weights
            uA = np.dot(XA, wA)  # compute equiangular direction
            a = X.T @ uA
            # if it's final step, when we have just added the last variable,
            # we must jump right to the full OLS solution to end the path
            if len(A) == N:
                gamma = C / AA
            # otherwise, calculate coefficient for update
            else:
                M_minus = (C - c) / (AA - a + eps)
                M_plus  = (C + c) / (AA + a + eps)
                M = np.nan_to_num(np.vstack((M_minus, M_plus)))
                M[M <= 0] = np.inf
                gamma = np.min(M)
                gamma = np.min([gamma, C/AA])

            if self.method == 'lasso':
                # maybe some feature(s) weight(s) cross zero - change weight sign after update.
                # it's situation for Lasso to zero this weight(s) and drop feature(s)
                # from the active set (until it comes back later).
                bad_signs_ind = []
                w_signs = np.sign(w[A])  # len = A
                w_update = gamma * np.multiply(wA, s)  # len = A
                sign_change = np.multiply(w_signs, np.sign(w[A] + w_update)) # len = A
                bad_signs_ind = [i for i, s in enumerate(sign_change) if s == -1]
                if bad_signs_ind:
                    bad_features_ind = [A[i] for i in bad_signs_ind] # indices in dim N
                    # calculate point, where weight crossed zero, and recalculate gamma.
                    # we need this new gamma to stop path now exactly at zero in bad weight.
                    zero_gammas = -w[bad_features_ind] / (np.multiply(wA, s)[bad_signs_ind] + eps)
                    zero_gamma = np.min(zero_gammas)
                    mu = mu + zero_gamma * uA  # update fit vector
                    w[A] = w[A] + zero_gamma * np.multiply(wA, s)  # update weights
                    w[bad_features_ind] = 0.
                    A = [i for i in A if i not in bad_features_ind]  # drop bad feature from active set
                    add = False
                    print(f'Step: {step}  Dropped: {bad_features_ind}')
                else:
                    mu = mu + gamma * uA  # update fit vector
                    w[A] = w[A] + w_update  # update weights
                    add = True
                    
            elif self.method == 'lars':
                mu = mu + gamma * uA  # update fit vector                
                w[A] = w[A] + gamma * np.multiply(wA, s)  # update weights
            
            self.alphas_.append(C)
            coef_path.append(w.tolist())
            step += 1
        
        C_ols = np.max(np.abs(np.dot(X.T, y - mu)))  # add last C (at OLS step)
        self.alphas_.append(C_ols)
        self.alphas_ = np.array(self.alphas_)
        self.alphas_ = self.alphas_ / X.shape[0]

        self.coef_ = w
        self.coef_path_ = np.array(coef_path)
    
    def predict(self, X):
        X = self.standardizer_X.transform(X)
        y_pred = np.dot(X, self.coef_)
        y_pred = self.standardizer_y.inverse_transform(y_pred)
        return y_pred


if __name__ == '__main__':
    '''
    X = np.array([[-1, 1], 
                  [0, 0], 
                  [1, 1]])
    y = np.array([-1.1111, 0, -1.1111])
    '''
    from sklearn.linear_model import Lars, LassoLars, lars_path
    from sklearn import datasets
    from smlib.linear.ols import LinearRegression
    
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    #boston = datasets.load_boston()
    #X, y = boston.data, boston.target
    
    X_std = StandardScaler().fit_transform(X)
    y_std = StandardScaler(with_std=False).fit_transform(y.reshape(-1, 1)).reshape((len(y)))
    alphas, _, sklars_coef_path_ = lars_path(X_std, y_std, method='lasso', verbose=3)
    
    xx = np.sum(np.abs(sklars_coef_path_.T), axis=1)
    xx /= xx[-1]
    
    plt.figure(figsize=(12, 8))
    plt.plot(xx, sklars_coef_path_.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('sklearn Path')
    plt.axis('tight')
    plt.show()
    
    
    ols = LinearRegression()
    ols.fit(X, y)
    #print(ols.coef_)
    
    lars = LARS(method='lasso')
    lars.fit(X, y)
    #print('check predict on Lars weights')
    #ny = 10
    #print(y[:ny])
    #print(ols.predict(X[:ny]))
    #print(lars.predict(X[:ny]))
    
    xx = np.sum(np.abs(lars.coef_path_), axis=1)
    xx /= xx[-1]
    
    plt.figure(figsize=(12, 8))
    plt.plot(xx, lars.coef_path_)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('smlib Path')
    plt.axis('tight')
    plt.show()
    
