from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import KFold
import heapq
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(subset='all',
                                         categories=['alt.atheism','sci.space'])

td_idf = TfidfVectorizer()
X = td_idf.fit_transform(newsgroups.data)
y = newsgroups.target
"""
grid = {'C': np.power(10.0,np.arange(-5,6))}
cv = KFold(n_splits=5,shuffle=True,random_state=241)
clf = SVC(kernel='linear',random_state=241)
gs = GridSearchCV(clf,grid,scoring='accuracy',cv=cv)
gs.fit(X,y)

for a in gs.grid_scores_:
    print(a.mean_validation_score)
    print(a.parameters)
    print("---------")
C = 1
"""
clf = SVC(kernel='linear',random_state=241,C=1)
clf.fit(X,y)

coef = np.abs(clf.coef_.data)
top10 = heapq.nlargest(10,range(len(coef)),coef.take)
top10_fullpath = clf.coef_.indices[top10]

f_names = td_idf.get_feature_names()
strings = []
for i in top10_fullpath:
    strings.append(f_names[i])

print([s+"," for s in sorted(strings)])