from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

X = pd.read_csv('close_prices.csv').loc[:,'AXP':'XOM']

pca = PCA(n_components=10)
pca.fit(X)

dj_index = pd.read_csv('djia_index.csv')['^DJI']

component_1 = pca.transform(X)[:,0]

print(list(pca.components_[0]).index(max(list(pca.components_[0]))))
