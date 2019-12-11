# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 00:06:26 2019

@author: Семен
"""
import numpy as np

class BinaryClassificationMetrics:
    """
    Calculates popular binary classification metrics using BINARY predictions.
    """
    def __init__(self, y_true, y_pred, 
                 pos_label=1, neg_label=0):
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert y_true.shape == y_pred.shape
        assert len(y_true.shape) == 1
        assert y_true.shape[0] > 1
        assert all(y in (neg_label, pos_label) for y in np.unique(y_true))
                
        self.y_true = y_true
        self.y_pred = y_pred
        self.pos_label = pos_label
        self.neg_label = neg_label
        
        self._calculate_confusion_matrix_cells()
        
    def _calculate_confusion_matrix_cells(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        for yt, yp in zip(self.y_true, self.y_pred):
            if yt == self.pos_label and yp == self.pos_label:
                self.tp += 1
            elif yt == self.pos_label and yp == self.neg_label:
                self.fn += 1
            elif yt == self.neg_label and yp == self.pos_label:
                self.fp += 1
            elif yt == self.neg_label and yp == self.neg_label:
                self.tn += 1
    
    @property
    def confusion_matrix(self):
        """
        Return confusion matrix for binary classification.
        Axes are as in sklearn convention:
                        y_pred
                 |   0    |    1    |
                 --------------------
               0 |  TN    |    FP   |
        y_true ----------------------
               1 |  FN    |    TP   |
        """
        return np.array([[self.tn, self.fp],
                         [self.fn, self.tp]])
        
    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else .0 
    
    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else .0
    
    def fbeta_measure(self, beta=1.):
        """
        (Weighted) harmonic mean of precision and recall.
        """
        P = self.precision
        R = self.recall
        if P + R == 0.0:
            return 0.0
        return (1 + beta**2) * P * R / (P * beta**2 + R)
    
    @property
    def f1(self):
        return self.fbeta_measure(beta=1.)
    
    @property
    def accuracy(self):
        return (self.tn + self.tp) / (self.tn + self.tp + self.fp + self.fn)
    
    @property
    def sensitivity(self):
        """
        Sensitivity == Recall for positive class == True Positive Rate (TPR)
        """
        return self.recall
    
    @property
    def specificity(self):
        """
        Specificity == Recall for negative class == True Negative Rate (TNR)
        """
        return self.tn / (self.tn + self.fp) if self.tn + self.fp > 0 else .0
        
    def zero_one_loss(self, normalize=True):
        """
        If normalize=True, return fracture of misclassifications (y_true != y_pred).
        Else, return number of misclassifications.
        """
        zero_one_loss = len((self.y_true - self.y_pred).nonzero()[0])
        if normalize:
            zero_one_loss /= len(self.y_pred)
        return zero_one_loss
    
    @property    
    def balanced_accuracy(self):
        """
        Average of per-class accuracies.
        """
        return 0.5 * (self.sensitivity + self.specificity)
    
"""
Stand-alone functions that calculate some binary classification metrics
using real-value model outputs (= scores = decision_function values).
"""

def roc_curve(y_true, y_score, pos_label=1):
    if np.unique(y_true).shape[0] == 1:
        raise ValueError('ROC is undefined if y_true contains only one class')

    tpr = [.0]
    fpr = [.0]
    thresholds = []
    sort_indices = np.argsort(-y_score)
    y_sorted = y_true[sort_indices]
    y_score_sorted = y_score[sort_indices]
    i, j = 0, 0
    M = len(y_score_sorted)
    tpr_step = 1 / len(y_true[y_true == pos_label])
    fpr_step = 1 / len(y_true[y_true != pos_label])
    while i < M:
        j = 0
        cur_t = y_score_sorted[i]
        while i + j < M and y_score_sorted[i + j] == cur_t:
            j += 1
        cur_y = y_sorted[i:i+j]
        n_tpr_steps = len(cur_y[cur_y == pos_label])
        n_fpr_steps = len(cur_y[cur_y != pos_label])
        cur_tpr = tpr[-1] + tpr_step * n_tpr_steps
        cur_fpr = fpr[-1] + fpr_step * n_fpr_steps
        tpr.append(cur_tpr)
        fpr.append(cur_fpr)
        thresholds.append(cur_t)
        i += j
    return np.array(fpr), np.array(tpr), thresholds
        
        
def roc_auc_score(y_true, y_score, pos_label=1):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    i = 1
    auc = 0.0
    while i < len(fpr):
        delta_fpr = fpr[i] - fpr[i-1]
        delta_tpr = tpr[i] - tpr[i-1]
        if delta_fpr > 0:
            auc += delta_fpr * tpr[i-1]
            # IMPORTANT CASE. When multiple objects in test set from different
            # classes have got the same equal value of model score ("threshold"), 
            # ROC makes diagonal piece of its path. So, we have to add the 
            # square of this triangle to the AUC.
            if delta_tpr > 0:
                auc += 0.5 * delta_fpr * delta_tpr
        i += 1
    return auc
            


if __name__ == '__main__':
    y = np.array([1, 1, 2, 2])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    pos_label = 2
    #y = np.array([1, 1, 1, 0, 1, 1, 0, 0])
    #scores = np.array([1., 1., 0.9, 0.6, 0.6, 0.4, 0.1, 0.05])

    fpr, tpr, thresholds = roc_curve(y, scores, pos_label)
    print(roc_auc_score(y, scores, pos_label))
