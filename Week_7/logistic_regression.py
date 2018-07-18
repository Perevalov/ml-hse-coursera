import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler


def l2_regression_first_test():
    #загружаем датасет
    features = pd.read_csv('features.csv',index_col='match_id')

    #получаем кол-во записей в датасете
    rows_num = features.shape[0]

    #заполняем пустоты в признаках
    for f,n in features.count().items():
        if n != rows_num:
            features[f].fillna(features[f].mean(),inplace=True)

    #формируем матрицы
    X = features.loc[:,'lobby_type':'dire_first_ward_time']
    y = features['radiant_win']

    #масштабируем признаки
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #инициализируем кросс-валидатор
    kf = KFold(n_splits=5,shuffle=True,random_state=42)

    #инициализируем грид с параметром
    grid = {'C': np.power(10.0, np.arange(-5, 5))}

    #инициализируем логистическую регрессию
    clf=LogisticRegression(random_state=241)

    #инициализируем поисковик параметра
    gs = GridSearchCV(clf,grid,scoring='roc_auc',cv=kf)

    #ставим отметку времени начала обучения
    start_time = datetime.datetime.now()
    gs.fit(X,y)

    #вычисляем потраченное время
    time_ = datetime.datetime.now() - start_time
    print(time_)

    for a in gs.grid_scores_:
        print(a.mean_validation_score,'__',a.parameters)

def l2_regression_no_categorical_features_test():
    # загружаем датасет
    features = pd.read_csv('features.csv', index_col='match_id')

    # получаем кол-во записей в датасете
    rows_num = features.shape[0]

    # заполняем пустоты в признаках
    for f, n in features.count().items():
        if n != rows_num:
            features[f].fillna(features[f].mean(), inplace=True)

    features_list = []

    for f in list(features.columns):
        if not any(f==i for i in ['start_time','lobby_type','radiant_win','duration']) \
                and 'hero' not in f and 'status' not in f:
            features_list.append(f)
            print(f)

    X = features.loc[:,features_list]
    y = features['radiant_win']

    # инициализируем кросс-валидатор
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # инициализируем грид с параметром
    grid = {'C': np.power(10.0, np.arange(-5, 5))}

    # инициализируем логистическую регрессию
    clf = LogisticRegression(random_state=241)

    # инициализируем поисковик параметра
    gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf)

    # ставим отметку времени начала обучения
    start_time = datetime.datetime.now()
    gs.fit(X, y)

    # вычисляем потраченное время
    time_ = datetime.datetime.now() - start_time
    print(time_)

    for a in gs.grid_scores_:
        print(a.mean_validation_score, '__', a.parameters)

l2_regression_no_categorical_features_test()



