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
    def __init__(self, y_true, y_pred, pos_label=1):
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert y_true.shape == y_pred.shape
        assert len(y_true.shape) == 1
        assert y_true.shape[0] > 1
        assert pos_label in y_true
                
        self.y_true = y_true
        self.y_pred = y_pred
        self.pos_label = pos_label
        
        self._calculate_confusion_matrix_cells()
        
    def _calculate_confusion_matrix_cells(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        for yt, yp in zip(self.y_true, self.y_pred):
            if yt == self.pos_label and yp == self.pos_label:
                self.tp += 1
            elif yt == self.pos_label and yp != self.pos_label:
                self.fn += 1
            elif yt != self.pos_label and yp == self.pos_label:
                self.fp += 1
            elif yt != self.pos_label and yp != self.pos_label:
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
    
    @property
    def mcc(self):
        """
        Matthew's Correlation Coefficient = sqrt(P1 * R1 * P0 * R0),
        ie. multiplication of geometric means of precision and recall 
        for every class.
        """
        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn
        return tp * tn / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    

"""
Stand-alone functions that calculate some binary classification metrics
using real-value model outputs (scores or probas or decision_function values).
"""

def roc_curve(y_true, y_score, pos_label=1):
    assert len(y_true) == len(y_score)
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
          
  
def log_loss(y_true, y_proba, pos_label=1, normalize=True, eps=1e-15):
    assert y_true.shape == y_proba.shape
    #  log-loss is undefined for p=0 or p=1, so we must clip
    y_proba_clip = np.clip(y_proba, eps, 1-eps)
    logloss = -np.sum([y * np.log(p) + (1 - y) * np.log(1-p) \
                       for y, p in zip(y_true, y_proba_clip)])
    if normalize:
        logloss /= len(y_true)
    return logloss
    

def precision_recall_curve(y_true, y_score, pos_label=1):
    assert len(y_true) == len(y_score)
    
    precisions = [1.]
    recalls = [0.]
    thresholds = []
    f1 = [0.]
    sort_indices = np.argsort(-y_score)
    y_sorted = y_true[sort_indices]
    y_score_sorted = y_score[sort_indices]
    i, j = 0, 0
    M = len(y_score_sorted)
    # start filling confusion matrix with threshold = max(y_score) + eps
    # i.e we start from predicting no positive class, only negative for all objects.
    tp = 0
    fp = 0
    tn = len(y_true[y_true != pos_label])
    fn = M - tn
    # getting threshold lower, incrementally changing confusion matrix
    while i < M:
        j = 0
        cur_t = y_score_sorted[i]
        while i + j < M and y_score_sorted[i + j] == cur_t:
            j += 1
        cur_y = y_sorted[i:i+j]
        for y in cur_y:
            if y == pos_label:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
        cur_p = tp / (tp + fp)
        cur_r = tp / (tp + fn)
        precisions.append(cur_p)
        recalls.append(cur_r)
        thresholds.append(cur_t)
        f1.append(2*cur_p*cur_r/(cur_p+cur_r) if (cur_p+cur_r)>0 else 0.0)
        # if we reached recall == 1, we cannot further improve metrics for
        # positive class, so no need to make threshold lower.
        if cur_r == 1.:
            break
        i += j
    # flip arrays to match scikit-learn implementation
    return  (np.array(precisions)[::-1], np.array(recalls)[::-1], 
             thresholds[::-1], np.array(f1)[::-1])


def pr_auc_score(y_true, y_score, pos_label=1):
    p, r, _ = precision_recall_curve(y_true, y_score, pos_label)
    return np.abs(np.trapz(p, r))


class MulticlassMetrics:
    def __init__(self, y_true, y_pred):
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred.shape) == 1, "By now, one hot multiclass labels format is not supported."
        assert y_true.shape == y_pred.shape
        assert len(np.unique(y_true)) > 1
        
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = dict()
        self.sup = dict()
        self.class_weights = dict()
        self.total = len(y_true)

        for c in np.unique(y_true):
            self.metrics[c] = BinaryClassificationMetrics(y_true, y_pred, pos_label=c)
            self.sup[c] = len(y_true[y_true==c])
            self.class_weights[c]  = self.sup[c] / self.total
            
    def classification_report(self):
        s = 'Label\t\tP\tR\tF1\tSupport\n'
        P = []
        R = []
        F1 = []
        P_w = []
        R_w = []
        F1_w = []
        for c in self.metrics:
            m = self.metrics[c]
            p = m.precision
            r = m.recall
            f1 = m.f1
            sup = self.sup[c]
            P.append(p)
            R.append(r)
            F1.append(f1)
            P_w.append(self.class_weights[c] * p)
            R_w.append(self.class_weights[c] * r)
            F1_w.append(self.class_weights[c] * f1)
            s += f'{c:d}\t\t{p:.2f}\t{r:.2f}\t{f1:.2f}\t{sup}\n'
        p_macro = np.mean(P)
        r_macro = np.mean(R)
        f1_macro = np.mean(F1)
        p_w = np.sum(P_w)
        r_w = np.sum(R_w)
        f1_w = np.sum(F1_w)
        s += f'\navg macro\t{p_macro:.2f}\t{r_macro:.2f}\t{f1_macro:.2f}\t{self.total}'
        s += f'\navg weighted\t{p_w:.2f}\t{r_w:.2f}\t{f1_w:.2f}\t{self.total}\n'
        return s
            
            

if __name__ == '__main__':
    y = np.array([1, 1, 2, 2])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    pos_label = 2
    #y = np.array([1, 1, 1, 0, 1, 1, 0, 0])
    #scores = np.array([1., 1., 0.9, 0.6, 0.6, 0.4, 0.1, 0.05])

    #fpr, tpr, thresholds = roc_curve(y, scores, pos_label)
    #print(roc_auc_score(y, scores, pos_label))

    precision, recall, thresholds, _ = precision_recall_curve(
        y, scores, pos_label=2)