import numpy as np

class kmedians:
    def __init__(self, data, k):

        self.n_data, self.n_dim = np.shape(data)
        self.k = k
        self.data = data

    def kmedians_train(self, max_iter=10):

        # Find the minimum and maximum values for each feature
        data_min = self.data.min(axis=0)
        data_max = self.data.max(axis=0)

        # Pick the centre locations randomly
        self.centers = np.random.rand(self.k, self.n_dim) * (data_max - data_min) + data_min
        old_centers = np.random.rand(self.k, self.n_dim) * (data_max - data_min) + data_min

        count = 0
        # print centres
        while np.sum(np.sum(old_centers - self.centers)) != 0 and count < max_iter:

            old_centers = self.centers.copy()
            count += 1
            if count == max_iter:
                print("Warning: maximum iteration reached.")

            # Compute distances
            distances = np.ones((1, self.n_data)) * np.sum((self.data - self.centers[0, :]) ** 2, axis=1)
            for j in range(self.k - 1):
                distances = np.append(distances,
                                      np.ones((1, self.n_data)) * np.sum((self.data - self.centers[j + 1, :]) ** 2, axis=1),
                                      axis=0)

            # Identify the closest cluster
            cluster = distances.argmin(axis=0)
            cluster = np.transpose(cluster * np.ones((1, self.n_data)))

            # Update the cluster centres
            for j in range(self.k):
                this_cluster = np.where(cluster == j)[0]
                if len(this_cluster) > 0:
                    self.centers[j, :] = np.median(self.data[this_cluster], axis=0)

        return self.centers
