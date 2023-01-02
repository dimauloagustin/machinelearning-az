import statsmodels.regression.linear_model as lm
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

def backwardElimination(x, y, sl):
    num_vars = len(x[0])
    temp = np.zeros((len(x), num_vars)).astype(int)
    for i in range(0, num_vars):
        regressor = lm.OLS(y, x.tolist()).fit()
        max_p = max(regressor.pvalues).astype(float)
        adj_r_before = regressor.rsquared_adj.astype(float)
        if(max_p > sl):
            for j in range(0, num_vars-i):
                if(regressor.pvalues[j].astype(float) == max_p):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, axis=1)
                    tmp_regressor = lm.OLS(y, x.tolist()).fit()
                    adj_r_after = tmp_regressor.rsquared_adj.astype(float)

                    if(adj_r_before >= adj_r_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, axis=1)
                        print(regressor.summary())
                        return x_rollback
    print(regressor.summary())
    return x

dataset = pd.read_csv('50_Startups.csv')
X: np.ndarray = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values


ct = ColumnTransformer(
    [('One hot state', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

X = X[:, 1:]

X = np.append(np.ones((len(X), 1)).astype(int), X, axis=1)

model = backwardElimination(X, Y, 0.05)

