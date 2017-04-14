import networkx as nx
import matplotlib.pyplot as plt


original_graph = nx.read_gml("polblogs.gml")
original_graph = original_graph.to_undirected()

print("Number of nodes: %s" % nx.number_of_nodes(original_graph))
print("Number of edges: %s" % nx.number_of_edges(original_graph))

ego_centric_networks = [];

for node in nx.nodes(original_graph):
    ego_graph = nx.ego_graph(original_graph, node)
    ego_centric_networks.append(ego_graph)



print(len(ego_centric_networks))
print(nx.number_of_nodes(ego_centric_networks[0]))
print(nx.number_of_edges(ego_centric_networks[0]))

nx.draw(ego_centric_networks[0])
nx.draw(ego_centric_networks[0], pos=nx.spectral_layout(ego_centric_networks[0]), nodecolor='r', edge_color='b')
plt.show()