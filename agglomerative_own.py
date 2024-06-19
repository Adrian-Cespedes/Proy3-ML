from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist, squareform
import numpy as np
import heapq

def get_distance_measure(M):
    if M == 'euclidean':
        return lambda cluster1, cluster2: np.linalg.norm(np.mean(cluster1, axis=0) - np.mean(cluster2, axis=0))
    return None

class AgglomerativeHierarchicalClustering:
    def __init__(self, data, K, M='euclidean'):
        self.data = data
        self.N = len(data)
        self.K = K
        self.measure = get_distance_measure(M)
        self.clusters = self.init_clusters()
        self.heap = self.init_heap()

    def init_clusters(self):
        return {i: [point] for i, point in enumerate(self.data)}

    def init_heap(self):
        dist_matrix = squareform(pdist(self.data, 'euclidean'))
        heap = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                heapq.heappush(heap, (dist_matrix[i, j], i, j))
        return heap

    def find_closest_clusters(self):
        while self.heap:
            distance, ci_id, cj_id = heapq.heappop(self.heap)
            if ci_id in self.clusters and cj_id in self.clusters:
                return ci_id, cj_id
        return None, None

    def merge_and_form_new_clusters(self, ci_id, cj_id):
        new_cluster_id = min(ci_id, cj_id)
        old_cluster_id = max(ci_id, cj_id)

        self.clusters[new_cluster_id] += self.clusters[old_cluster_id]
        del self.clusters[old_cluster_id]

        new_distances = []
        for cluster_id in self.clusters:
            if cluster_id != new_cluster_id:
                new_distance = self.measure(self.clusters[new_cluster_id], self.clusters[cluster_id])
                heapq.heappush(self.heap, (new_distance, min(new_cluster_id, cluster_id), max(new_cluster_id, cluster_id)))

    def run_algorithm(self):
        while len(self.clusters) > self.K:
            ci_id, cj_id = self.find_closest_clusters()
            if ci_id is None or cj_id is None:
                break
            self.merge_and_form_new_clusters(ci_id, cj_id)

        # print(f"Número final de clusters: {len(self.clusters)}")

    def get_cluster_labels(self):
        labels = np.zeros(self.N)
        for cluster_id, points in self.clusters.items():
            for point in points:
                point_idx = np.where(np.all(self.data == point, axis=1))[0][0]
                labels[point_idx] = cluster_id
        
        
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = np.array([label_mapping[label] for label in labels])
        
        return labels

    def print_clusters(self):
        for cluster_id, points in self.clusters.items():
            print(f"Cluster: {cluster_id}")
            for point in points:
                print(f"    {point}")