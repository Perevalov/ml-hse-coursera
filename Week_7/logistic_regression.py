import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

#процедура подбора параметра С для логистической регрессии
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

    #выводим полученные значения по точности и соответствующее значение параметра
    for a in gs.grid_scores_:
        print(a.mean_validation_score,'__',a.parameters)

#процедура подбора параметра С для логистической регрессии без категориальных признаков в датасете
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
    #заполняем массив наименований признаков
    for f in list(features.columns):
        if not any(f==i for i in ['start_time','lobby_type','radiant_win','duration']) \
                and 'hero' not in f and 'status' not in f:
            features_list.append(f)
            print(f)

    #формируем матрицы
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

    #выводим полученные значения качества и соответствующие параметры
    for a in gs.grid_scores_:
        print(a.mean_validation_score, '__', a.parameters)

#процедура вычисления уникального количества героев в датасете
def count_unique_heros():
    # загружаем датасет
    features = pd.read_csv('features.csv', index_col='match_id')

    # получаем кол-во записей в датасете
    rows_num = features.shape[0]

    # заполняем пустоты в признаках
    for f, n in features.count().items():
        if n != rows_num:
            features[f].fillna(features[f].mean(), inplace=True)

    values = []
    #заполняем массив наименований признаков
    for f in list(features.columns):
        if 'hero' in f:
            values.append(f)

    #решейпим матрицу в вектор
    df1 = pd.lreshape(features,{'hero':values})

    #выводим кол-во уникальных значений (героев)
    print(df1['hero'].value_counts().shape[0])

#процедура тестирования логистической регрессии на "мешке слов№
def bag_of_words_test():
    #искусственно увеличим размер массива (т.к. в датасете имеются пропуски)
    N = 112

    #считываем данные
    features = pd.read_csv('features.csv', index_col='match_id')

    #инициализируем нулевую матрицу заданного размера (мешок слов)
    X_pick = np.zeros((features.shape[0],N))

    #заполняем мешок слов
    for i,match_id in enumerate(features.index):
        for p in range(5):
            X_pick[i, features.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, features.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

    values = []
    #заполняем массив наименований признаков
    for f in list(features.columns):
        if not any(f==i for i in ['start_time','match_id','radiant_win','duration']) \
                and 'hero' not in f and 'status' not in f:
            values.append(f)

    #инициализируем матрицу с "нужными" признаками
    X_ = features.loc[:, values]

    #соединяем 2 датафрейма
    X = pd.DataFrame(np.hstack((X_.values,X_pick)))

    #инициализируем целевой вектор
    y = features['radiant_win']

    #получаем кол-во записей в датасете
    rows_num = features.shape[0]

    # заполняем пустоты в признаках
    for f, n in X.count().items():
        if n != rows_num:
            X[f].fillna(X[f].mean(), inplace=True)

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

    # выводим полученные значения качества и соответствующие параметры
    for a in gs.grid_scores_:
        print(a.mean_validation_score, '__', a.parameters)




