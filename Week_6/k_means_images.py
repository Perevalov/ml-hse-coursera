from skimage.io import imread
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage import img_as_float
from skimage.measure import compare_psnr

image = imread('parrots.jpg')
matrix = np.reshape(image,(image.shape[0]*image.shape[1],3))

clf = KMeans(init='k-means++',random_state=241,n_clusters=11)
clf.fit(matrix)

mean_matr = matrix.copy()
median_matr = matrix.copy()

for i in range(clf.n_clusters):
    mean_matr[clf.labels_ == i] = np.mean(mean_matr[clf.labels_ == i],axis=0)

for i in range(clf.n_clusters):
    median_matr[clf.labels_ == i] = np.median(median_matr[clf.labels_ == i],axis=0)

print(compare_psnr(matrix,median_matr))
print(compare_psnr(matrix,mean_matr))
