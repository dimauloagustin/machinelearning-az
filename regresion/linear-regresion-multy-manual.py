import statsmodels.regression.linear_model as lm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X: np.ndarray = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values


ct = ColumnTransformer(
    [('One hot state', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

regresion = LinearRegression()
regresion.fit(X_train, Y_train)

y_pred = regresion.predict(X_test)

# añadimos una columna que represente el termino independiente
X = np.append(np.ones((len(X), 1)).astype(int), X, axis=1)

SL = 0.05
x_opt: np.ndarray = X[:, [0, 1, 2, 3, 4, 5]]

regresion_OLS = lm.OLS(endog=Y, exog=x_opt.tolist()).fit()
regresion_OLS.summary()

x_opt: np.ndarray = X[:, [0, 1, 3, 4, 5]]

regresion_OLS = lm.OLS(endog=Y, exog=x_opt.tolist()).fit()
regresion_OLS.summary()

x_opt: np.ndarray = X[:, [0, 3, 4, 5]]

regresion_OLS = lm.OLS(endog=Y, exog=x_opt.tolist()).fit()
regresion_OLS.summary()

x_opt: np.ndarray = X[:, [0, 3, 5]]

regresion_OLS = lm.OLS(endog=Y, exog=x_opt.tolist()).fit()
regresion_OLS.summary()
