import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter
    

dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-' \
                      'databases/00396/Sales_Transactions_Dataset_Weekly.csv')

print("Apresentando o shape dos dados (dimenssoes)")
print(dataset.shape)

print("")
print(dataset[['Product_Code', 'W0', 'W51', 'MIN', 'MAX', 'Normalized 0',
               'Normalized 51']].dtypes)

print("\nVisualizando o conjunto inicial (head) dos " \
      "dados, ou mais claramente, os 20 primeiros registros (head(20))")
print(dataset.head(20))

print("\nConhecendo os dados estatisticos dos dados carregados (describe)")
print(dataset.describe())

print("\nNomes das colunas")
print(dataset.columns)

# drop 'Product_Code' e W1, W2... W50, W51
columns_to_drop = ['Product_Code']
for n in range(0, 52):
    columns_to_drop.append('W' + str(n))
    
dataset.drop(columns_to_drop, 1, inplace=True)

X = dataset.values

wcss = []
wcss2 = []
maxit = 11
 
for i in range(1, maxit):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans2 = KMeans(n_clusters = i)
    kmeans.fit(X)
    kmeans2.fit(X)
    wcss.append(kmeans.inertia_)
    wcss2.append(kmeans2.inertia_)
plt.plot(range(1, maxit), wcss, 'bo', c='red') 
plt.plot(range(1, maxit), wcss2, 'g-', c='blue')
plt.show()

X = dataset.values

# Initializing KMeans
kmeans = KMeans(n_clusters=3)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
print(labels)

colors = []
for label in labels:
    if label == 0:
        colors.append('orange')
    elif label == 1:
        colors.append('green')
    else:
        colors.append('cyan')

# Getting the cluster centers
C = kmeans.cluster_centers_

plt.rcParams['figure.figsize'] = (16, 9)

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=150)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='red', s=1000)
plt.show()

X = dataset.values

# criamos o objeto para realizar o agrupamento aglomerativo
Hclustering = AgglomerativeClustering(n_clusters=3, affinity="euclidean", 
                                      linkage="ward")
Hclustering.fit(X)

# Getting the cluster labels
labels = Hclustering.fit_predict(X)

colors = []
for label in labels:
    if label == 0:
        colors.append('cyan')
    elif label == 1:
        colors.append('green')
    else:
        colors.append('orange')

silhouette_avg = silhouette_score(X, labels)
print("For n_clusters =", 3, 
      "The average silhouette_score is :", silhouette_avg)

fig = plt.figure()
ax = Axes3D(fig)
plt.title('Agglomerative clustering')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=150)

plt.show()

print("Padrao de vendas dos clusters")
for i in range(3):
    
    color = ''
    
    if i == 0:
        color = 'orange'
    elif i == 1:
        color = 'green'
    else:
        color = 'cyan'
    
    plt.plot(range(1, 53), C[i, 2:], 'bo', c=color)
    
    plt.title('Cluster ' + str(i+1))
    plt.xlabel('Semanas')
    plt.ylabel('Vendas') #within cluster sum of squares
    plt.show()
