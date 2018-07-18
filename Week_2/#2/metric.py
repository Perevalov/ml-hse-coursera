from sklearn import datasets
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

dataset = datasets.load_boston()

X = dataset.data
y = dataset.target
X = preprocessing.scale(X)

range_ = np.linspace(1,10,num=200)

scores = []
kf = KFold(n_splits=5,random_state=42,shuffle=True)

for p in range_:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    neigh.fit(X, y)
    scores.append(cross_val_score(neigh,X,y,cv=kf,scoring='neg_mean_squared_error').mean())

print(scores)
print(scores.index(max(scores)))
