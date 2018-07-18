import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('data.csv',header=None)

X = data[[1,2]]
y = data[[0]]

clf = SVC(C=100000,random_state=241,kernel='linear')
clf.fit(X,y)

print(clf.support_)