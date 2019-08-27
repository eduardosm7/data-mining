#Disciplina: Solucoes de Mineracao de dados
#--------------------------------------------------------
#Script para a analise exploratoria dos dados (AED)
#--------------------------------------------------------


# Importando as bibliotecas necessarias
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Fazendo o carregamento dos dados diretamente do UCI Machine Learning
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

url="http://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"

# Definindo o nome de cada coluna dos dados
names = ['age','sex','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','class']
dataset = pandas.read_csv(url, names=names)

print("Apresentando o shape dos dados (dimenssoes)")
print(dataset.shape)

print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente"
		"os 20 primeiros registros (head(20))")
print(dataset.head(20))

print("Conhecendo os dados estatisticos dos dados carregados (describe)")
print(dataset.describe())

print("Conhecendo a distribuicao dos dados por classes (class distribution)")
print(dataset.groupby('class').size())

print("Criando grafios de caixa da distribuicao das classes")
dataset.plot(kind='box', subplots=True, layout=(17,17), sharex=False, sharey=False)
plt.show()

print("Criando histogramas dos dados por classes")
dataset.hist()
plt.show()

print("Criando graficos de dispersao dos dados")
scatter_matrix(dataset)
plt.show()