import networkx as nx
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt


# original_graph = nx.read_gml("polblogs.gml")
# original_graph = original_graph.to_undirected()
# print(original_graph.number_of_nodes())
# print(original_graph.number_of_edges())
# adj = nx.to_numpy_matrix(original_graph)
#
#
# spectral = cluster.SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors")
# spectral.fit(adj)
# y_pred = spectral.labels_.astype(np.int)
#
# print(y_pred)

original_graph = nx.read_gml("polblogs.gml")
original_graph = original_graph.to_undirected()

print("Number of nodes: %s" % nx.number_of_nodes(original_graph))
print("Number of edges: %s" % nx.number_of_edges(original_graph))

ego_centric_networks = []
ego_centric_networks_clusters = []

spectral = cluster.SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors")

for node in nx.nodes(original_graph):
    ego_graph = nx.ego_graph(original_graph, node)
    ego_centric_networks.append(ego_graph)

    if nx.number_of_nodes(ego_graph) > 10:
        ego_adj = nx.to_numpy_matrix(ego_graph)
        spectral.fit(ego_adj)
        y_pred = spectral.labels_.astype(np.int)
        ego_centric_networks_clusters.append(y_pred)
    else:
        ego_centric_networks_clusters.append('r')

# nx.draw(ego_centric_networks[0])
for i in range(0, 20):
    nx.draw(ego_centric_networks[i], pos=nx.draw_circular(ego_centric_networks[i]),
            node_color=ego_centric_networks_clusters[i], edge_color='b')
    plt.show()
