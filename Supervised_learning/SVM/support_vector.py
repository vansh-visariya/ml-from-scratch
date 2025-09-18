# in SVM we use the hyperplane to separate the data into two classes
# the hyperplane is defined by the weights and the bias, the distance between the hyperplane and the closest data point is called the margin = 2/||w||

# for the loss we use hinge loss, which is max(0, 1 - y(w.x + b))
# this loss is 0 if the data point is correctly classified and on the correct side of the margin, 
# otherwise it is increasing positive as far as the data point is from the margin

# loss function is loss = lambda * (1/2 * ||w||^2) + (1/n * sum(max(0, 1 - y(w.x + b))))
# lambda is the regularization parameter, it controls the trade-off between the margin and the misclassification error (so we can have a large margin and still classify all the points correctly)
# ||w|| is the magnitude of the weight vector, it is the distance from the origin to the hyperplane

# use kernel for increasing the dimensionality of the data
# this is done by mapping the data to a higher dimensional space using a kernel function
# eg. kernel(x, x') = (x.x' + 1)^2
# this helps to find a hyperplane that separates the data in the higher dimensional space for better seperation
import numpy as np

# def linear_kernel(x1, x2):
#     return np.dot(x1, x2)

# def polynomial_kernel(x1, x2, p=3):
#     return (1 + np.dot(x1, x2))**p

# def rbf_kernel(x1, x2, gamma=0.1):
#     """ 
#     Radial Basis Function (Gaussian) kernel. Maps to infinite dimensions.
#     Gamma controls the 'reach' of a single training example.
#     """
#     distance_sq = np.sum((x1 - x2)**2)
#     return np.exp(-gamma * distance_sq)

class SVM:
    def __init__(self, lr=0.01, lambda_param = 0.01, kernel = None, gamma = None, n_iters=100):
        """
        lr: learning rate
        lambda_param: regularization parameter
        n_iters: number of iterations
        kernel: kernel function (linear, polynomial, rbf)
        gamma: parameter for the rbf kernel
        """
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.gamma = gamma

        if self.kernel is None:
            self.kernel = "linear"
        else:
            self.kernel = kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y<=0, -1, 1) # convert to -1 and 1

        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # check the condition where the data point is correctly classified and on the correct side of the margin
                # based on which we update the weights and the bias
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    # if they are correctly classified, the loss is 0, so we only update the weights
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                    # bias is not updated
                else:
                    # if they are misclassified, the loss depends on the distance from the margin
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.lr * y[idx]

    def predict(self, X):
        y_pred = np.dot(X, self.weights) - self.bias
        y_pred = np.where(y_pred<=0, 0, 1) # convert to 0 and 1
        return y_pred
