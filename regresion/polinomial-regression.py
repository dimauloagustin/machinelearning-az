import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X: np.ndarray = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(X)
regresion = LinearRegression()
regresion.fit(x_poly, Y)

plt.scatter(X, Y, color="red")
plt.plot(X, regresion.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Polynomial Regresion")
plt.xlabel("Position")
plt.ylabel("Salary")