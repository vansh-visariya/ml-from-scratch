# PCA finds a new set of dimensions such that all the deminsions are linearly independent 

# why we what is reduce the dimensionality of the data? 
# answer :-  to remove the noise from the data, to remove the redundant information from the data, 
# to make the data easier to visualize, to make the data easier to process, to make the data easier to store

# but how does PCA reduce the dimensionality of the data? 
# answer :- PCA reduces dimensions by projecting the data onto the directions (principal components) where the variance is highest, 
# keeping the most important information while reducing noise

# key note:
# 1. high variance in the data means there can be good info but also at the same time there can be outliers also
# 2. high variance model is bad (supervised learning), they learn the outliers also, and we don't want that

# covariance matrix? -> The covariance matrix is a square matrix that shows the covariance (relationship) 
# between every pair of features in your dataset.

# eigenvectors ? -> are the directions in which data varies the most (principal components)
# eigenvalues ? -> are the amount of variance in the data in those directions (eigenvectors)
# just for curiosity, how to solve eigenvalue problem? 
# Solve the characteristic equation: det(A−λI)=0 where λ is the eigenvalue and I is the identity matrix.
# for each λ, solve the system of linear equations: (A−λI)v=0 where v is the eigenvector.

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components # number of principal components we want to keep
        self.components = None
        self.mean = None
    
    def fit(self, x):
        self.mean = np.mean(x, axis=0) # mean of each feature
        x = x - self.mean # center the data
        # PCA's goal is to find directions that maximize variance. If one feature has a much larger
        # scale than others (e.g., salary in thousands vs. age in years), it will dominate the
        # variance calculation and the resulting principal components. Standardizing ensures that
        # each feature contributes equally to the analysis. We only center the data (subtract mean)
        # here because the covariance calculation is not affected by scaling, but centering is crucial.

        # The formula is (1/N) * (X^T . X), where N is the number of samples.
        cov = np.cov(x.T) # We use X_std.T because np.cov expects features as rows.

        eigenvalues, eigenvectors = np.linalg.eig(cov) 
        # - Eigenvectors: These are the directions of the axes of the new feature space.
        #   They are vectors that, when multiplied by the covariance matrix, only get scaled, not
        #   changed in direction. This means they are the "principal axes" of variance in the data.
        # - Eigenvalues: These are scalars that tell us the amount of variance captured by each
        #   eigenvector (principal component). A high eigenvalue means that eigenvector captures
        #   a lot of variance in the data.

        eigenvectors = eigenvectors.T 
        idxs = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[idxs]

        self.components = sorted_eigenvectors[:self.n_components]


    def transform(self, x):
        X_std = (x - self.mean)
        projected_data = np.dot(X_std, self.components.T)
        # The dot product is a way of projecting one vector onto another. By taking the dot product
        # of our standardized data (n_samples, n_features) with our components matrix
        # (n_components, n_features), we are essentially calculating the "coordinates" of our data
        # points in the new coordinate system defined by the principal components. This transforms
        # the data from its original feature space to the new, lower-dimensional space.
        return projected_data