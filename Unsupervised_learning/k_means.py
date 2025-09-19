# how k means works?
# 1. we randomly select k points as our initial centroids
# 2. we assign each point to the nearest centroid
# 3. we move the centroids to the mean of the points assigned to them
# 4. we repeat steps 2 and 3 until the centroids stop moving

import numpy as np

def euclidean_distance(x1, x2):
    # formula :- sqrt((x1 - x2)^2 + (y1 - y2)^2)
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans:
    def __init__(self, k=5, max_iter=100):
        self.k = k
        self.max_iter = max_iter
    
    def predict(self, x):
        self.x = x
        self.n_samples, self.n_features = x.shape

        # initialize centroids with random samples
        random_sample_idx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[i] for i in random_sample_idx]

        # optimize clusters
        for _ in range(self.max_iter):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            # if clusters have not changed, we have found the optimal clusters
            if self._is_converged(centroids_old, self.centroids):
                break
        
        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)
    
    def _create_clusters(self, centroids):
        self.clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x):
            centroid_idx = self._closest_centroid(sample, centroids)
            self.clusters[centroid_idx].append(idx)
        return self.clusters
    
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # get mean of all points in a cluster
        centroids = np.zeros((self.k, self.n_features)) # use tuples for 2d array
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for clusters_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = clusters_idx
        return labels