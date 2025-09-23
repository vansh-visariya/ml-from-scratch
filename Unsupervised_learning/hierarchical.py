# Agglomerative Hierarchical Clustering implemented from scratch using Python and NumPy.

# steps involved in hierarchical clustering? 
# 1. consider each data point to be seperate clusters
# 2. find the nearest point and merge them into one cluster
# 3. repeat step 2 until all the points are merged into one cluster

# how do we define the "distance" between two clusters?
# 1. single linkage -> minimum distance between two points in two clusters
# 2. complete linkage -> maximum distance between two points in two clusters
# 3. average linkage -> average distance between two points in two clusters 
import numpy as np
from scipy.cluster.hierarchy import dendrogram # to plot the dendrogram
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

class HierarchicalClustering:
    def __init__(self):
        # now only using complete linkage
        self.linkage_matrix = None
    
    def _calculate_cluster_distance(self, cluster1_points, cluster2_points):
        max_dist = 0.0
        # For 'complete' linkage, the distance between two clusters is defined as the distance
        # between the two FARTHEST points across the two clusters. We iterate through every
        # possible pair of points (one from each cluster) to find this maximum distance.
        for p1 in cluster1_points:
            for p2 in cluster2_points:
                dist = euclidean_distance(p1, p2)
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    def fit(self, X):
        n_samples, _ = X.shape
        
        # Each cluster is a dictionary holding its unique ID and the indices of the points it contains.
        clusters = [{'id': i, 'points_indices': [i]} for i in range(n_samples)]
        next_cluster_id = n_samples
        self.linkage_matrix = []
        # The algorithm starts by treating every single data point as its own cluster.
        # We assign a unique ID (0 to n_samples-1) to each initial cluster. When we merge
        # clusters later, we will create new clusters with new IDs, starting from `n_samples`.
        # The `linkage_matrix` will store the history of every merge we perform.

        # We repeat the merging process until only one cluster remains.
        while len(clusters) > 1:
            min_dist = float('inf')
            closest_pair_indices = None

            # We iterate through every unique pair of clusters to find the one with the minimum distance.
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]
                    
                    # Get the actual data points for each cluster to calculate distance
                    points1 = X[cluster1['points_indices']]
                    points2 = X[cluster2['points_indices']]
                    
                    dist = self._calculate_cluster_distance(points1, points2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_pair_indices = (i, j)
            
            # This is the core of the algorithm. At each iteration, we must find the two most
            # similar (closest) clusters in the current set of clusters. This pair will be
            # the next to be merged.

            idx1, idx2 = closest_pair_indices
            
            # Ensure idx1 is smaller than idx2 for consistent removal later
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1

            cluster1 = clusters[idx1]
            cluster2 = clusters[idx2]

            # The format is [cluster_id_1, cluster_id_2, distance, num_points_in_new_cluster].
            # This format is compatible with SciPy's dendrogram function.
            num_new_points = len(cluster1['points_indices']) + len(cluster2['points_indices'])
            self.linkage_matrix.append([cluster1['id'], cluster2['id'], min_dist, num_new_points])

            # Hierarchical clustering is not just about the final grouping; it's about the
            # entire history of merges. The linkage matrix is that history. It tells us
            # which clusters were merged, at what distance, and how many points were in the
            # resulting new cluster. This is exactly what's needed to draw the dendrogram.

            # Create the new merged cluster and update the list
            new_points_indices = cluster1['points_indices'] + cluster2['points_indices']
            new_cluster = {'id': next_cluster_id, 'points_indices': new_points_indices}
            next_cluster_id += 1
            
            # Remove the old clusters (remove the one with the higher index first).
            clusters.pop(idx2)
            clusters.pop(idx1)
            
            # Add the new merged cluster.
            clusters.append(new_cluster)
            # This is the "agglomeration" step. We create a new cluster that contains all the
            # points from the two old clusters. We give it a new ID and then update our list
            # of active clusters by removing the two that were just merged and adding the new one.
            # The loop then repeats with this updated list of clusters.

        # Convert the linkage matrix to a NumPy array for compatibility with SciPy.
        self.linkage_matrix = np.array(self.linkage_matrix)
    

    def plot_dendrogram(self):
        if self.linkage_matrix is None:
            raise Exception("Please fit the model first before plotting the dendrogram.")
            
        plt.figure(figsize=(10, 7))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        dendrogram(self.linkage_matrix)
        plt.show()