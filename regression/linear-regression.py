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


# Fazendo o carregamento dos dados diretamente do UCI Machine Learning
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
      "cpu-performance/machine.data"

# Definindo o nome de cada coluna dos dados
names = ['vendor name','Model Name','MYCT','MMIN','MMAX','CACH','CHMIN',
         'CHMAX','PRP','ERP']
dataset = pandas.read_csv(url, names=names)

print("Descrevendo a base de dados")
print(dataset.describe())

print("")
print(dataset.shape)
print("")
print(dataset.dtypes)
print("")

dataset.drop(columns=['Model Name'], inplace=True)
dataset.drop(columns=['ERP'], inplace=True)
dataset.drop(columns=['vendor name'], inplace=True)

print(dataset.corr())

# Load the hardware computer dataset
X = dataset.values[:,:-1]
Y = dataset.values[:,-1]

#Split data into training an test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, 
                                                    random_state=0)

#features
hardware_X_train = X_train[:,2:3]
hardware_X_test = X_test[:,2:3]
hardware_y_train = y_train
hardware_y_test = y_test

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(hardware_X_train, hardware_y_train)

# Make predictions using the testing set
hardware_y_pred = regr.predict(hardware_X_test)

# The mean squared error
print("\nRoot Mean squared error: %.2f"
      % sqrt(mean_squared_error(hardware_y_test, hardware_y_pred)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(hardware_y_test, hardware_y_pred))

# Plot outputs
plt.scatter(hardware_X_test, hardware_y_test,  color='red')
plt.plot(hardware_X_test, hardware_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
