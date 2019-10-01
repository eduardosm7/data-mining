# -*- coding: utf-8 -*-

# Importando as bibliotecas necessarias
import pandas
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler



# Fazendo o carregamento dos dados diretamente do UCI Machine Learning
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"

# Definindo o nome de cada coluna dos dados
names = ['vendor name','Model Name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
dataset = pandas.read_csv(url, names=names)

print("Apresentando o shape dos dados (dimenssoes)")
print(dataset.shape)

print("Apresentando o tipo das colunas gerado pelo read_csv")
print(dataset.dtypes)

dataset.drop(columns=['Model Name'], inplace=True)
dataset.drop(columns=['ERP'], inplace=True)
dataset.drop(columns=['vendor name'], inplace=True)

print(dataset.corr())

# Load the hardware computer dataset
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]


#Split data into training an test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.values.reshape(-1, 1)
y_test  = y_test.values.reshape(-1, 1)

y_scaler = StandardScaler()
# Fit on training set only.
y_scaler.fit(y_train)

# Apply transform to both the training set and the test set.
y_train = y_scaler.transform(y_train)
y_test = y_scaler.transform(y_test)

model = linear_model.LinearRegression()
model.fit(X_train,y_train)


# Make predictions using the testing set
hardware_y_pred = model.predict(X_test)

# The mean squared error
print("\nRoot Mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, hardware_y_pred)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, hardware_y_pred))

# Plot outputs
plt.scatter(y_test, hardware_y_pred,  color='red')
#plt.plot(X_test, hardware_y_pred, color='blue', linewidth=3)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=4)


plt.xticks(())
plt.yticks(())

plt.show()
