import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import time
import datetime

#загружаем датасет
features = pd.read_csv('features.csv',index_col='match_id')

#получаем кол-во записей в датасете
rows_num = features.shape[0]

#заполняем пустоты в признаках
for f,n in features.count().items():
    if n != rows_num:
        features[f].fillna(99999,inplace=True)
        print(features[f])

#формируем матрицы
X = features.loc[:,'lobby_type':'dire_first_ward_time']
y = features['radiant_win']

#инициализируем массив с кол-вом деревьев
n_trees = [30]
kf = KFold(n_splits=5,shuffle=True,random_state=42)
scores = []

start_time = datetime.datetime.now()
for n in n_trees:
    clf = GradientBoostingClassifier(n_estimators=n,random_state=241,verbose=True)
    clf.fit(X,y)
    scores.append(cross_val_score(clf,X,y,cv=kf,scoring='roc_auc').mean())

time_ = datetime.datetime.now() - start_time

print(scores)
print(time_)









