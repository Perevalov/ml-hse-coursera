import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

data = pd.read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data.loc[:,'Sex':'ShellWeight']
y = data['Rings']

scores = []
kf = KFold(n_splits=5,random_state=1,shuffle=True)

for i in range(1,50):
    clf = RandomForestRegressor(random_state=1,n_estimators=i)
    clf.fit(X, y)
    scores.append(cross_val_score(clf, X, y, cv=kf, scoring='r2').mean())

for s in scores:
    if s > 0.52:
        print(s)
        print(scores.index(s))
print(scores)

