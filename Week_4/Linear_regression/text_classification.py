import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import re
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

#считываем датасет
data = pd.read_csv("salary-train.csv")
test_data = pd.read_csv("salary-test-mini.csv")

#предобрабатываем текстовый признак
text = data['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))
test_text = test_data['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))

#векторизируем с помощью TD-IDF
td_idf = TfidfVectorizer(min_df=5)
text_vectorized = td_idf.fit_transform(text)
test_text_vectorized = td_idf.transform(test_text)

#заполняем пропущенные значения как НАН
data['LocationNormalized'].fillna('nan',inplace=True)
data['ContractTime'].fillna('nan',inplace=True)

test_data['LocationNormalized'].fillna('nan',inplace=True)
test_data['ContractTime'].fillna('nan',inplace=True)

#делаем one-hot кодирование категориальных признаков
enc = DictVectorizer()
location_contract = enc.fit_transform(data[['LocationNormalized','ContractTime']].to_dict('records'))
test_location_contract = enc.transform(test_data[['LocationNormalized','ContractTime']].to_dict('records'))

#склеиваем признаки
X_train = sparse.hstack([text_vectorized,location_contract])
X_test = sparse.hstack([test_text_vectorized,test_location_contract])
y_train = data['SalaryNormalized']

clf = Ridge(alpha=1.0,random_state=241)
clf.fit(X_train,y_train)

for p in clf.predict(X_test):
    print(np.round(p,2))


