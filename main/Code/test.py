import generalized_spectral_clustering as gsc
import networkx as nx
import helpers as h

graph = nx.read_gml("../Data/karate.gml")

gsc_model = gsc.gsc(graph, 3)
node_list, clusters = gsc_model.get_clusters(kmedian_max_iter=100, num_cluster_per_node=False)

print(h.get_cluster_overlap(clusters))

print(h.get_cluster_coefficient(clusters))
#print(node_list)
#print(clusters)

# hashtable = {2: [6, 4], 5: [9, 14], 1: [2, 9]}
# hashtable2 = {2: [6, 4], 5: [9, 14], 0: [4, 0]}
#
# a, b = h.keep_common_keys(hashtable, hashtable2)
# print(len(a))
# print(b)
#
# for key in hashtable:
#     print(type(key))