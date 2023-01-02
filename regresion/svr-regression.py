import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X: np.ndarray = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_preprocesed = scaler_x.fit_transform(X)
y_preprocesed = scaler_y.fit_transform(Y.reshape(-1,1)).reshape(-1,)

regresion = SVR(kernel="rbf")
regresion.fit(x_preprocesed, y_preprocesed)

y_pred =scaler_y.inverse_transform(regresion.predict(scaler_x.transform([[6.5]])).reshape(-1, 1)).reshape(-1,)

plt.scatter(X, Y, color="red")
plt.plot(X, scaler_y.inverse_transform(regresion.predict(x_preprocesed).reshape(-1, 1)).reshape(-1,), color="blue")
plt.title("SVR Regresion")
plt.xlabel("Position")
plt.ylabel("Salary")