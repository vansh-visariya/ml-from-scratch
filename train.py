import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_reg.linear_regression import LinearRegression
from sklearn import datasets
from residue.error import mean_squared_error

x, y = datasets.make_regression(n_samples=1000, n_features=3, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

regressor = LinearRegression(lr=0.01, n_iters=1000)
regressor.fit(x_train, y_train)
predictions = regressor.predict(x_test)

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)


## plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, c='b', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], c='r', label='Actual')
plt.xlabel("Actual")
plt.ylabel("Predictions")
plt.title("Actual vs Predictions")
plt.legend()
plt.show()