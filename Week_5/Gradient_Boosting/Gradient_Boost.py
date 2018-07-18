import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

def plot(test_loss,train_loss):
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

def forest():
    data = pd.read_csv('gbm-data.csv')

    X = data.loc[:, 'D1':'D1776']
    y = data['Activity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

    clf = RandomForestClassifier(n_estimators=37,random_state=241)
    clf.fit(X_train,y_train)

    y_pred = clf.predict_proba(X_test)
    print(np.round(log_loss(y_test,y_pred),2))

def gradient():
    data = pd.read_csv('gbm-data.csv')

    X = data.loc[:,'D1':'D1776']
    y = data['Activity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,random_state=241)

    learning_rates =  [1, 0.5, 0.3, 0.2, 0.1]


    clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=250,verbose=True,random_state=241)
    clf.fit(X_train,y_train)

    log_loss_train = []
    log_loss_test = []

    for y_pred in clf.staged_decision_function(X_train):
        log_loss_train.append(log_loss(y_train,1/(1+np.exp(-y_pred))))

    for y_pred in clf.staged_decision_function(X_test):
        log_loss_test.append(log_loss(y_test,1/(1+np.exp(-y_pred))))

    plot(log_loss_test,log_loss_train)

forest()


