import pandas as pd
from sklearn import metrics

def max_precision(precision,recall):
    precision_arr = []
    for i in range(len(precision)):
        if recall[i] >= 0.7:
            precision_arr.append(precision[i])

    return max(precision_arr)

data = pd.read_csv("scores.csv")
y_true = data[['true']]
log_reg = data[['score_logreg']]
svm = data[['score_svm']]
knn = data[['score_knn']]
tree = data[['score_tree']]
"""
print(metrics.roc_auc_score(y_true,log_reg))
print(metrics.roc_auc_score(y_true,svm))
print(metrics.roc_auc_score(y_true,knn))
print(metrics.roc_auc_score(y_true,tree))
"""

precision,recall, thresholds = metrics.precision_recall_curve(y_true,log_reg)
print(max_precision(precision,recall))

precision,recall, thresholds = metrics.precision_recall_curve(y_true,svm)
print(max_precision(precision,recall))

precision,recall, thresholds = metrics.precision_recall_curve(y_true,knn)
print(max_precision(precision,recall))

precision,recall, thresholds = metrics.precision_recall_curve(y_true,tree)
print(max_precision(precision,recall))



