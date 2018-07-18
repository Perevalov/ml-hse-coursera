from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
import pandas as pd
import numpy as np

#считываем данные из файла
data = pd.read_csv("../wine.data")

#инициализируем матрицу признаков и вектор ответов
X = data[['1','2','3','4','5','6','7','8','9','10','11','12','13']]
#проводим масштабирование признаков
X = preprocessing.scale(X)
y = data['0']

#инициализируем кросс-валидатор
kf = KFold(n_splits=5,random_state=42,shuffle=True)
scores = []

#тестируем на различных классификаторах оценку кросс валидации
for i in range(1,50):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X, y)
    scores.append(cross_val_score(neigh,X,y,cv=kf,scoring='accuracy').mean())

#выводим индекс элемента с макисмальным значением кросс-валидации
print(scores.index(max(scores)))
#выводим макисмальное значение кросс-валидации
print(np.round(max(scores),2))
