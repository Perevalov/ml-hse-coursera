import  pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def m():
    data = pd.read_csv("data.csv", header=None)
    X = np.array(data[[1,2]])
    y = data[[0]]
    w_l2 = l2_regression(data,C=10)
    score = []
    for row in X:
        #делаем предсказания
        score.append(logistic_regression([row[0], row[1]],w_l2))

    #считаем скор
    print(np.round(roc_auc_score(y,score),3))

    w_l = l2_regression(data)
    score = []
    for row in X:
        score.append(logistic_regression([row[0], row[1]], w_l))

    print(np.round(roc_auc_score(y, score),3))

#настраиваем веса
def l2_regression(data, C=0):
    l = len(data)
    i = 0
    k=0.1
    res = 100.0
    w = np.array([0.0, 0.0])
    w_tmp = np.array([0.0, 0.0])


    while i <= 10000:

        w_tmp[0] = w[0]
        w_tmp[1] = w[1]

        sum1 = data.apply(
            lambda row: row[0] * row[1] * (1 - sigmoid(row[0] * w.dot(row[1:3]))), axis=1
        )
        w[0] = w[0] + k/l * np.sum(sum1) - k*C*w[0]

        sum2 = data.apply(
            lambda row: row[0] * row[2] * (1 - sigmoid(row[0] * w.dot(row[1:3]))), axis=1
        )
        w[1] = w[1] + k/l*np.sum(sum2) - k*C*w[1]

        res = math.sqrt((w_tmp[0]-w[0])**2+(w_tmp[1]-w[1])**2)

        if (res <= 10**-5):
            break
        i+=1

    return w

def logistic_regression(x,w):
    return 1 / (1 + math.exp(-w[0]*x[0] - w[1]*x[1]))

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

m()