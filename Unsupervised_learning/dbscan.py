# steps to implement/solve the DBSCAN
# 1. Pick an unvisited data point
# 2. Find all points within the eps(hyperparameter) distance of this point
# 3. if the point has more than min_samples(hyperparameter) neighbors, add them to the cluster(core point)
#    if the point has less than min_samples neighbors, mark it as noise(outlier), later can be border point
# 4. repeat all the steps for the unvisited points

import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise
    """
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        self.X = None
        self.labels = None
    
    def _get_neighbors(self, point_index):
        """
        Finds all points in the dataset that are within the `eps` distance
        of a given point.
        """
        neighbors = []
        # This is the first step in classifying any point (as core, border, or noise).
        # We need to identify its local neighborhood by checking its distance to every
        # other point in the dataset.
        for other_point_index in range(len(self.X)):
            if euclidean_distance(self.X[point_index], self.X[other_point_index]) <= self.eps:
                neighbors.append(other_point_index)
        return neighbors
    
    def _expand_cluster(self, point_index, neighbors, cluster_id):
        """
        Expands a cluster starting from a core point.
        """
        # This function is called ONLY when we've identified a core point. Its job is
        # to find every single point that is "density-reachable" from that starting point.

        # Assign the cluster ID to the starting core point.
        self.labels[point_index] = cluster_id
        
        # We use a list as a queue (First-In, First-Out) to explore the neighborhood. BFS
        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]
            
            # If this neighbor was previously marked as NOISE, it's now a border point.
            # We assign it to the current cluster.
            if self.labels[neighbor_index] == -1: # -1 indicates NOISE
                self.labels[neighbor_index] = cluster_id

            # If the neighbor has not been visited yet (is still labeled 0),
            # we explore it further.
            elif self.labels[neighbor_index] == 0: # 0 indicates UNVISITED
                self.labels[neighbor_index] = cluster_id
                
                # Find the neighbors of this neighbor
                new_neighbors = self._get_neighbors(neighbor_index)
                
                # If this neighbor is ALSO a core point, its neighborhood must also be
                # part of the cluster. We add its neighbors to our queue to be processed.
                if len(new_neighbors) >= self.min_pts:
                    neighbors.extend(new_neighbors)
            
            i += 1
            # This is the "chain reaction" part of DBSCAN. By finding the neighbors of neighbors
            # (if they are core points), we can connect dense regions that are close to each
            # other, allowing the algorithm to find non-spherical, arbitrarily shaped clusters.
        
    def fit_predict(self, X):
        """
        Performs DBSCAN clustering and returns the cluster labels.
        """
        self.X = X
        n_samples = X.shape[0]
        
        # labels: 0 = unvisited, -1 = noise, >0 = cluster ID
        self.labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0

        # We need a way to track the state of every point.
        # Starting with 0 (unvisited)
        # 1 is the standard label for noise/outliers in DBSCAN.
        # We will increment `cluster_id` for each new cluster we discover.

        # We iterate through every single point in the dataset.
        for point_index in range(n_samples):
            
            if self.labels[point_index] != 0:
                continue
            
            # If a point was visited as part of a previous cluster expansion, its label will no
            # longer be 0. Skipping it prevents redundant work and ensures we only try to
            # start new clusters from points that haven't been claimed yet.

            neighbors = self._get_neighbors(point_index)
            
            # If it doesn't have enough neighbors, it's currently noise.
            if len(neighbors) < self.min_pts:
                self.labels[point_index] = -1 # Mark as NOISE
            else:
                # It's a core point, so we create a new cluster ID.
                cluster_id += 1
                # And expand the cluster starting from this point.
                self._expand_cluster(point_index, neighbors, cluster_id)
        
        return self.labels 