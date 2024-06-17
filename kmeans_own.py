import numpy as np
import matplotlib.pyplot as plt

class MyKMeans:
    def __init__(self, data, k, umbral):
        np.random.seed(2024)
        self.data = data
        self.k = k
        self.umbral = umbral

    def distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    # def Init_Centroide(self, data, k):
    #     indices = np.random.choice(data.shape[0], size=k, replace=False)
    #     return data[indices]

    def Init_Centroide(self, data, k):
        # Inicialización de centroides utilizando K-means++
        print("using ++")
        centroids = np.zeros((k, data.shape[1]))
        centroids[0] = data[np.random.choice(data.shape[0])]  # Seleccionar el primer centroide aleatoriamente

        for i in range(1, k):
            distances = np.array([min([np.linalg.norm(centroid - point) for centroid in centroids]) for point in data])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()

            for j, cumulative_probability in enumerate(cumulative_probabilities):
                if r < cumulative_probability:
                    centroids[i] = data[j]
                    break

        return centroids

    def return_new_centroide(self, grupos, data, k):
        new_centroids = np.zeros((k, data.shape[1]))

        for i in range(k):
            cluster_points = data[grupos == i]

            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = 0

        return new_centroids

    def get_cluster(self, data, centroides):
        labels = np.zeros(data.shape[0], dtype=int)
        for i in range(data.shape[0]):
            distances = np.zeros(self.k)
            for j in range(self.k):
                distances[j] = np.linalg.norm(data[i] - centroides[j])
            labels[i] = np.argmin(distances)
        return labels

    def distancia_promedio_centroides(self, old_centroides, new_centroides):
        promedios = []

        for i in range(old_centroides.shape[0]):
            dist = self.distance(old_centroides[i], new_centroides[i])
            promedios.append(dist)

        return np.mean(promedios)

    def kmeans(self):
        centroides =  self.Init_Centroide(self.data, self.k)
        clusters   =  self.get_cluster(self.data, centroides)
        new_centroides = self.return_new_centroide(clusters, self.data, self.k)

        while(self.distancia_promedio_centroides(centroides, new_centroides) > self.umbral):
            centroides = new_centroides
            clusters   =  self.get_cluster(self.data, centroides)
            new_centroides = self.return_new_centroide(clusters, self.data, self.k)

        return new_centroides, clusters

# data = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [7, 8]])
# print(data.shape)
# k = 2
# umbral = 0.05
# mykm = MyKMeans(data, k, umbral)
# centroides, clusters = mykm.kmeans()
# print(clusters)