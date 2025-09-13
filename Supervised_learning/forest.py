import numpy as np
from Supervised_learning.decision_tree import DecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = [] # list of decision trees
    
    def fit(self, X, y):
        # create n_trees number of decision trees
        for _ in range(self.n_trees):
            tree = DecisionTree(self.max_depth, self.min_samples_split, self.n_features)
            x_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    # this gives us the random sample of the data
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs] 

    def predict(self, X):
        # for each sample we use n_trees to predict
        prediction = np.array([tree.predict(X) for tree in self.trees])
        # prediction.shape = (n_trees, n_samples)
        prediction = np.swapaxes(prediction, 0, 1)
        # swap the axes to get the prediction for each sample
        y_pred = np.array([self._most_common_label(pred) for pred in prediction])
        return np.array(y_pred)
