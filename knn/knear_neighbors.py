# In knn, we calucalte distance between two points and find the nearest neighbors 
# and classify the point based on the majority class of the k neighbors

# in knn data should be normalized, bcs of the distance is calculated should be on same scale
import numpy as np
from collections import Counter # to count the number of labels

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self.pred(x) for x in X]
        return np.array(y_pred)
    
    def pred(self, x):
        # get the distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest labels
        k_idx = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_idx]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1) # returns a list of tuples (label, count) 
        # most_common(1) returns the one most common label
        return most_common[0][0]