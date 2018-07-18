import pandas as pd
import  numpy as np
from sklearn import metrics


data = pd.read_csv("classification.csv")

y_true = data[['true']]
y_pred = data[['pred']]

TP = 0
FP = 0
FN = 0
TN = 0

for index,row in data.iterrows():
    if row[0] == 1 and row[1] == 1:
        TP += 1
    elif row[0] == 0 and row[1] == 1:
        FP += 1
    elif row[0] == 1 and row[1] == 0:
        FN += 1
    elif row[0] == 0 and row[1] == 0:
        TN +=1

print(str(TP)+" "+str(FP)+" "+str(FN)+" "+str(TN))

accuracy = metrics.accuracy_score(y_true=y_true,y_pred=y_pred)
precision = metrics.precision_score(y_true=y_true,y_pred=y_pred)
recall = metrics.recall_score(y_true=y_true,y_pred=y_pred)
f_measure = metrics.f1_score(y_true=y_true,y_pred=y_pred)

print(str(np.round(accuracy,2))+" "+str(np.round(precision,2))+" "+str(np.round(recall,2))+" "+str(np.round(f_measure,2)))
