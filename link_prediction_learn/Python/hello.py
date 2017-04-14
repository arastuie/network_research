import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_node(1)

G.add_nodes_from([2, 3, 4, 5, 6])

G.add_edge(1, 2)

e = (2, 3)

G.add_edge(*e)

G.add_edges_from([(1, 3), (3, 4), (1, 5), (2, 6)])

#print(G.number_of_edges())
#print(G.nodes())

#print(G.number_of_nodes())
#print(G.edges())

#print(G.neighbors(3))

G2 = nx.read_gml("dolphins.gml")

#print(G2.number_of_edges())
#print(G2.number_of_nodes())

#print(G2.nodes())

#print(nx.shortest_path(G2, source = 'kalblog.com', target = 'rogerailes.blogspot.com'))
#print(nx.shortest_path(G2, source = 'kalblog.com', target = 'selfish-souls.com/blog'))
print(nx.adjacency_matrix(G2))

nx.adamic_adar_index(G2);
#nx.draw_circul(G2)
#plt.show()