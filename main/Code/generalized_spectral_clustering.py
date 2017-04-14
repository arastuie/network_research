import networkx as nx
import numpy as np
import scipy as sp
import kmedians as kmed

class gsc:
    def __init__(self, graph, K):
        self.node_list = graph.nodes()
        self.A = nx.to_numpy_matrix(graph, nodelist=self.node_list)
        self.K = K

    def get_clusters(self, kmedian_max_iter=100, num_cluster_per_node=False):
        n_dim = np.shape(self.A)[0]

        # Getting K smallest eigenvalues and eigenvectors of the adj matrix
        w, U = sp.sparse.linalg.eigs(self.A, k=self.K, which='LR', return_eigenvectors=True)
        L = np.diag(np.real(w))
        U = np.real(U)

        # computing the estimated latent node positions
        X = np.dot(U, L ** 0.5)

        # computing values of alpha and tau
        alpha = (np.sum(self.A)) / (n_dim * (n_dim - 1) * self.K)
        tau = 0.1 * (alpha ** 0.2 * self.K ** 1.5) / n_dim ** 0.3

        # normalizing and regularizing X
        X_norm = (X / (np.linalg.norm(X, axis=1, keepdims=True) + tau))

        # running K-medians on X_norm
        km = kmed.kmedians(X_norm, self.K)
        S = km.kmedians_train(max_iter=kmedian_max_iter)

        # projecting rows of X_norm onto snap S
        # Z = np.dot(np.dot(X_norm, np.transpose(S)), np.linalg.pinv(np.dot(S, np.transpose(S))))
        Z = np.dot(X_norm, np.linalg.inv(S))

        # normalizing output z
        Z_norm = Z / np.linalg.norm(Z, axis=1, keepdims=True)

        # thresholding at 1/k to get hard membership
        hard_cluster = np.where(Z_norm < (1 / self.K), 0, 1)

        if num_cluster_per_node:
            hard_cluster = np.sum(hard_cluster, axis=1)

        return self.node_list, hard_cluster
