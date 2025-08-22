import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape ## (rows, columns)
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            ## gradient give us the direction of the steepest ascent
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))  ## there we removed 2 from the dw and db, bcs it just a constant
            ## X.T = (n_features, n_samples), (y_pred - y) = (n_samples, 1)
            ## dw = (n_features, 1)
            ## in dot product, inner dimensions must match
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw  ## we go opposite direction of the gradient
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred