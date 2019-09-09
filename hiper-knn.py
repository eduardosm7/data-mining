#Disciplina: Solucoes de Mineracao de dados
#--------------------------------------------------------
#Script para a analise exploratoria dos dados (AED)
#--------------------------------------------------------

# Importando as bibliotecas necessarias
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors

# Fazendo o carregamento dos dados diretamente do UCI Machine Learning
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
      "thyroid-disease/allhyper.data"

# Definindo o nome de cada coluna dos dados
names = [
    'age', 'sex', 'on thyroxine', 'query on thyroxine',
    'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
    'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
    'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
    'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source', 'class'
]

# Ler o CSV e ja altera valores de "?" por np.NaN
dataset = pandas.read_csv(url, names=names, na_values="?")

# Remove informação irrelevante depois da classe
dataset.replace(to_replace=r"negative\.\|\d*",
                value="negative",
                regex=True,
                inplace=True)
dataset.replace(to_replace=r"T3 toxic\.\|\d*",
                value="T3 toxic",
                regex=True,
                inplace=True)
dataset.replace(to_replace=r"goitre\.\|\d*",
                value="goitre",
                regex=True,
                inplace=True)
dataset.replace(to_replace=r"hyperthyroid\.\|\d*",
                value="hyperthyroid",
                regex=True,
                inplace=True)

#fazendo interpolacao para encontrar os novos valores para NaN
dataset = dataset.interpolate()

# print("Apresentando o tipo das colunas gerado pelo read_csv")
# print(dataset.dtypes)

# Remove a coluna TBG, já que não possui nenhum valor
dataset.drop(columns=['TBG'], inplace=True)

dataset = pandas.get_dummies(dataset, columns=['referral source'])
cols = dataset.columns.tolist()
cols.remove("class")
cols.append("class")
dataset = dataset[cols]

# Substitui outlier da idade pela mediana
median = dataset.loc[dataset['age'] <= 120, 'age'].median()
dataset["age"] = np.where(dataset["age"] > 120, median, dataset['age'])

#substitui todos os registros com o marcado ? para NaN
# print("Apresenta a contagem de valores NaN em cada coluna")
# print(dataset.isnull().sum())

dataset.fillna(method='ffill', inplace=True)

# print(dataset.isnull().sum())

# print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente," \
#       "os 20 primeiros registros (head(20))")
# print(dataset.head(20))

#Convertendo dados categóricos para dados numéricos
le = LabelEncoder()
for column_name in dataset.columns:
    if dataset[column_name].dtype == object:
        dataset[column_name] = le.fit_transform(dataset[column_name])
    else:
        pass

# print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente," \
#       "os 20 primeiros registros (head(20))")
# print(dataset.head(20))

# print("Apresentando o tipo das colunas gerado pelo read_csv")
# print(dataset.dtypes)

X = dataset.values[:, 0:-1]
Y = dataset['class']

params = [
    {
        "metric": "manhattan",
        "n_neighbors": 7
    },
    {
        "metric": "manhattan",
        "n_neighbors": 123
    },
    {
        "metric": "manhattan",
        "n_neighbors": 333
    },
    {
        "metric": "euclidean",
        "n_neighbors": 7
    },
    {
        "metric": "euclidean",
        "n_neighbors": 123
    },
    {
        "metric": "euclidean",
        "n_neighbors": 333
    },
    {
        "metric": "chebyshev",
        "n_neighbors": 7
    },
    {
        "metric": "chebyshev",
        "n_neighbors": 123
    },
    {
        "metric": "chebyshev",
        "n_neighbors": 333
    },
    {
        "metric": "minkowski",
        "n_neighbors": 7
    },
    {
        "metric": "minkowski",
        "n_neighbors": 123
    },
    {
        "metric": "minkowski",
        "n_neighbors": 333
    },
]

#usando o metodo para fazer uma unica divisao dos dados
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=10)

for param in params:
    clf = neighbors.KNeighborsClassifier(metric=param["metric"],
                                         n_neighbors=param["n_neighbors"])

    clf = clf.fit(X_train, y_train)

    print("============ Version {} ============".format(params.index(param)))

    print("Acuracia de trainamento clf: %0.3f" % clf.score(X_train, y_train))
    print("Acuracia de teste clf: %0.3f" % clf.score(X_test, y_test))
