#Disciplina: Solucoes de Mineracao de dados
#--------------------------------------------------------
#Script para a analise exploratoria dos dados (AED)
#--------------------------------------------------------

# Importando as bibliotecas necessarias
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

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
dataset.replace(
    to_replace=r"negative\.\|\d*", value="negative", regex=True, inplace=True)
dataset.replace(
    to_replace=r"T3 toxic\.\|\d*", value="T3 toxic", regex=True, inplace=True)
dataset.replace(
    to_replace=r"goitre\.\|\d*", value="goitre", regex=True, inplace=True)
dataset.replace(
    to_replace=r"hyperthyroid\.\|\d*",
    value="hyperthyroid",
    regex=True,
    inplace=True)

#fazendo interpolacao para encontrar os novos valores para NaN
dataset = dataset.interpolate()

print("Apresentando o tipo das colunas gerado pelo read_csv")
print(dataset.dtypes)

# Remove a coluna TBG, já que não possui nenhum valor
dataset.drop(columns=['TBG'], inplace=True)

# Substitui outlier da idade pela mediana
median = dataset.loc[dataset['age'] <= 120, 'age'].median()
dataset["age"] = np.where(dataset["age"] > 120, median, dataset['age'])

#substitui todos os registros com o marcado ? para NaN
print("Apresenta a contagem de valores NaN em cada coluna")
print(dataset.isnull().sum())

dataset.fillna(method='ffill', inplace=True)

print(dataset.isnull().sum())

print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente," \
      "os 20 primeiros registros (head(20))")
print(dataset.head(20))

#Convertendo dados categóricos para dados numéricos
le = LabelEncoder()
for column_name in dataset.columns:
    if dataset[column_name].dtype == object:
        dataset[column_name] = le.fit_transform(dataset[column_name])
    else:
        pass
    
print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente," \
      "os 20 primeiros registros (head(20))")
print(dataset.head(20))

print("Apresentando o tipo das colunas gerado pelo read_csv")
print(dataset.dtypes)

#divisão de dados atributos e classe
#X = dataset.values[:, 0:7]
#Y = dataset.values[:,8]

X = dataset.values[:, 0:28]
Y = dataset['class']

#usando o metodo para fazer uma unica divisao dos dados
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.25, 
                                                    random_state = 10)

#criando diferentes arvores
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf2 = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
clf2 = clf2.fit(X_train, y_train)

print("Acuracia de trainamento clf: %0.3f" %  clf.score(X_train, y_train))
print("Acuracia de teste clf: %0.3f" %  clf.score(X_test, y_test))

print("Acuracia de trainamento clf2: %0.3f" %  clf2.score(X_train, y_train))
print("Acuracia de teste clf2: %0.3f" %  clf2.score(X_test, y_test))

print("Profundidade das arvores criadas")
print(clf.tree_.max_depth)
print(clf2.tree_.max_depth)


# Constroi um classificador com arvore de decisao
x = dataset.values[:, 0:28]
y = dataset['class']

dt = tree.DecisionTreeClassifier(criterion='entropy')
dt.fit(x,y)
dotfile = open("hiper.dot", 'w')
tree.export_graphviz(dt, out_file=dotfile, feature_names=[
        'age',
        'sex',
        'on thyroxine',
        'query on thyroxine',
        'on antithyroid medication',
        'sick',
        'pregnant',
        'thyroid surgery',
        'I131 treatment',
        'query hypothyroid',
        'query hyperthyroid',
        'lithium',
        'goitre',
        'tumor',
        'hypopituitary',
        'psych',
        'TSH measured', 
        'TSH',
        'T3 measured',
        'T3',
        'TT4 measured',
        'TT4', 
        'T4U measured', 
        'T4U', 
        'FTI measured', 
        'FTI',
        'TBG measured', 
        'referral source'])
dotfile.close()
print("Arvore de decisao gerada no diretorio!")

