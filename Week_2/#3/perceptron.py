import numpy as np
from sklearn.linear_model import Perceptron
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv("train.csv",header=None)
data_test = pd.read_csv("test.csv",header=None)

X_train = data_train[[1,2]]
X_test = data_test[[1,2]]

y_train = data_train[[0]]
y_test = data_test[[0]]

clf = Perceptron(random_state=241)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

accuracy_before = metrics.accuracy_score(y_test,predictions)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled,y_train)
predictions = clf.predict(X_test_scaled)
accuracy_after = metrics.accuracy_score(y_test,predictions)

print(str(accuracy_before)+'____'+str(accuracy_after))
print(accuracy_after-accuracy_before)