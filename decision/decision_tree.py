# in this decision tree we are using information gain (ID3) as the criteria for splitting the tree

# stopping criteria :-
# maximum deapth , minimum number of samples , min impurity decrease

import numpy as np
from collections import Counter

class Node:
    def __init__(self,left=None, right= None, feature=None, threshold=None, value=None):
        self.left = left
        self.right = right
        self.feature = feature # how many feature the node is spliting on
        self.threshold = threshold # value of the feature at which the node is spliting
        self.value = value # value is passed only to the leaf node
    
    def is_leaf_node(self):
        return self.value is not None # tells if the leaf node has the value

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features # The number of features to consider at each split. 
        # If it's less than the total number of features, 
        # it introduces randomness that can improve model robustness (a core idea behind Random Forests).
        self.root = None
    
    def fit(self, X, y):
        # check for number of feature if it exceeds the total number of features
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    # helper function
    def _grow_tree(self, X, y, depth=0): # recursive function
        n_samples, n_features = X.shape 
        n_labels = len(np.unique(y))

        # stopping criteria
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # finding the best split
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False) 
        # this will return the index of the feature to be split, which introduces randomness
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:,best_feature], best_threshold)

        # checks if the split is pure
        if len(left_idxs)==0 or len(right_idxs)==0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(left, right, best_feature, best_threshold)
    
    def _best_split(self, X, y, feature_idxs):
        best_gain = -1

        # thershold_idx and idx are the value of the feature and the index of the feature at which the split is done
        threshold_idx, split_idx = None, None

        for id in feature_idxs:
            x_col = X[:,id]
            thresholds = np.unique(x_col) # all the unique values in that feature's column

            for thr in thresholds:
                # calculate the information gain for each thresholds 
                gain = self._information_gain(y, x_col, thr)

                if gain>best_gain:
                    best_gain = gain
                    threshold_idx = thr
                    split_idx = id
        return split_idx, threshold_idx
    
    def _information_gain(self, y, x_col, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create child nodes
        left_idxs, right_idxs = self._split(x_col, threshold)
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        
        # calculate the weighted avg. entropy of child nodes
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        # information gain for this split
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y) # count the number of unique values in y
        # example if np.bincount([1,2,3,1,2,1,1,2]) = [0, 4, 3, 1]

        ps = hist / len(y) # probability of each unique value
        return -np.sum([p * np.log(p) for p in ps if p>0])
    
    def _split(self, x_col, split_thresh):
        left_idxs = np.argwhere(x_col<=split_thresh).flatten()
        right_idxs = np.argwhere(x_col>split_thresh).flatten()
        # argwhere returns the index of the values that satisfy the condition
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature]<=node.threshold:
            # go to left node
            return self._traverse_tree(x, node.left)
        # go to right node
        return self._traverse_tree(x, node.right)