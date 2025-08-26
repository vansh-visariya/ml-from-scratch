import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_reg.linear_regression import LinearRegression
from linear_reg.elastic import ElasticNet
from linear_reg.lasso import LassoRegression
from linear_reg.ridge import RidgeRegression
from sklearn import datasets
from residue.error import *

x, y = datasets.make_regression(n_samples=1000, n_features=3, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = [LinearRegression(learning_rate=0.1, n_iters=1000), ElasticNet(lambda1_param=0.01, lambda2_param=0.01), LassoRegression(lambda1_param=0.01), RidgeRegression(lambda2_param=0.01)]

for model in models:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print("model:", model.__class__.__name__)
    print("r2_score:", r2)
    print("mean_squared_error:", mse)