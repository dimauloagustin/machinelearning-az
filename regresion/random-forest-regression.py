import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X: np.ndarray = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

regression = RandomForestRegressor(n_estimators=300,random_state=0)
regression.fit(X, Y)

y_pred = regression.predict([[6.5]])

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X, Y, color="red")
plt.plot(X_grid, regression.predict(X_grid), color="blue")
plt.title("Decission tree Regresion")
plt.xlabel("Position")
plt.ylabel("Salary")