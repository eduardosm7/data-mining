#Disciplina: Solucoes de Mineracao de dados
#--------------------------------------------------------
#Script para a analise exploratoria dos dados (AED)
#--------------------------------------------------------


# Importando as bibliotecas necessarias
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import random_projection

# Fazendo o carregamento dos dados diretamente do UCI Machine Learning
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"

# Definindo o nome de cada coluna dos dados
names = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','class']
dataset = pandas.read_csv(url, names=names)

print("Apresentando o shape dos dados (dimenssoes)")
print(dataset.shape)

print("Apresentando o tipo das colunas gerado pelo read_csv")
print(dataset.dtypes)

#substitui todos os registros com o marcado ? para NaN
dataset[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']] = dataset[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']].replace('?', np.NaN)
print("Apresenta a contagem de valores NaN em cada coluna")
print(dataset.isnull().sum())

#Abordagems para substituicao do NaN

#usando a media (mean) mediana (median) coluna por coluna
#dataset['A2'].fillna(dataset['A2'].median(), inplace=True)

#preenchendo com as ocorrencias mais próximas
#dataset.fillna(method='ffill',inplace=True)

#convertendo os dados com NaN para float
convert_dict = {'A2':float,'A3':float,'A8':float,'A11':int,'A14':float,'A15':int}
dataset=dataset.astype(convert_dict)
print("Apresentando os novos tipos das colunas")
print(dataset.dtypes)

#fazendo interpolacao para encontrar os novos valores para NaN
dataset=dataset.interpolate()
print("Mostra a quantidade de valores ausentes (NaN) de cada coluna")
print(dataset.isnull().sum())

#Analisando com PCA a varianca dos dados em relacao aos atributos, apenas para valores numericos
pca = PCA().fit(dataset[['A2','A3','A8','A11','A14','A15']])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente, os 20 primeiros registros (head(20))")
print(dataset.head(20))

print("Conhecendo os dados estatisticos dos dados carregados (describe)")
print(dataset.describe())

print("Conhecendo a distribuicao dos dados por classes (class distribution)")
print(dataset.groupby('class').size())

print("Criando graficos de caixa da distribuicao das classes")
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

print("Criando histogramas dos dados por classes")
dataset.hist()
plt.show()

print("Criando graficos de dispersao dos dados com paleta de cores")
colors_palette = {'+': 'red', '-': 'yellow'}
colors = [colors_palette[c] for c in dataset['class']]
scatter_matrix(dataset, c=colors)

plt.show()