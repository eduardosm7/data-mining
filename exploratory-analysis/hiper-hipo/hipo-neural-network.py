#Disciplina: Solucoes de Mineracao de dados
#--------------------------------------------------------
#Script para a analise exploratoria dos dados (AED)
#--------------------------------------------------------

# Importando as bibliotecas necessarias
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

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
dataset.replace(to_replace=r"primary hypothyroid\.\|\d*",
                value="primary hypothyroid",
                regex=True,
                inplace=True)
dataset.replace(to_replace=r"compensated hypothyroid\.\|\d*",
                value="compensated hypothyroid",
                regex=True,
                inplace=True)
dataset.replace(to_replace=r"secondary hypothyroid\.\|\d*",
                value="secondary hypothyroid",
                regex=True,
                inplace=True)
dataset.replace(to_replace=r"hypothyroid\.\|\d*",
                value="hypothyroid",
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

# {
#     "solver": [
#         "lbfgs", "sgd", "adam"
#     ],
#     "hidden_layer_sizes": (
#         number_of_layers,
#         number_of_neurons_per_layer
#     ),
#     "activation": [
#         "identity", "logistic", "tanh", "relu"
#     ]
# }

params = [
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 3),
        "activation": "identity",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 3),
        "activation": "identity",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 30),
        "activation": "identity",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 30),
        "activation": "identity",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 3),
        "activation": "logistic",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 3),
        "activation": "logistic",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 30),
        "activation": "logistic",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 30),
        "activation": "logistic",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 3),
        "activation": "tanh",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 3),
        "activation": "tanh",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 30),
        "activation": "tanh",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 30),
        "activation": "tanh",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 3),
        "activation": "relu",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 3),
        "activation": "relu",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 30),
        "activation": "relu",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 30),
        "activation": "relu",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 3),
        "activation": "identity",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 3),
        "activation": "identity",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 30),
        "activation": "identity",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 30),
        "activation": "identity",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 3),
        "activation": "logistic",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 3),
        "activation": "logistic",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 30),
        "activation": "logistic",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 30),
        "activation": "logistic",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 3),
        "activation": "tanh",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 3),
        "activation": "tanh",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 30),
        "activation": "tanh",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 30),
        "activation": "tanh",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 3),
        "activation": "relu",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 3),
        "activation": "relu",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 30),
        "activation": "relu",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 30),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 3),
        "activation": "identity",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 3),
        "activation": "identity",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 30),
        "activation": "identity",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "identity",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 3),
        "activation": "logistic",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 3),
        "activation": "logistic",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 30),
        "activation": "logistic",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "logistic",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 3),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 3),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 30),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 3),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 3),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 30),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "relu",
    },
]

#usando o metodo para fazer uma unica divisao dos dados
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=10)

for param in params:
    clf = MLPClassifier(solver=param["solver"],
                        hidden_layer_sizes=param["hidden_layer_sizes"],
                        activation=param["activation"],
                        random_state=10)

    clf = clf.fit(X_train, y_train)

    #print("============ Version {} ============".format(params.index(param)))

    # print("Acuracia de trainamento clf: %0.3f" % clf.score(X_train, y_train))
    # print("Acuracia de teste clf: %0.3f" % clf.score(X_test, y_test))
    print("%0.3f,%0.3f" %
          (clf.score(X_train, y_train), clf.score(X_test, y_test)))
