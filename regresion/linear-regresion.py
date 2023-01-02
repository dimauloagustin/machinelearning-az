from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/home/agufa/source/AtoZIA/regresion/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0)

regresion = LinearRegression()
regresion.fit(X_train, Y_train)

y_pred = regresion.predict(X_test)

plt.scatter(X_train, Y_train, color='red')
plt.scatter(X_test, Y_test, color='green')
plt.plot(X_train, regresion.predict(X_train), color='blue')
plt.title('Sueldos contra años de experiencia')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldos')
plt.show()
