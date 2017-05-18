import pickle
import helpers as h
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import GSC.generalized_spectral_clustering as gsc

# # reading FB graph
# original_graph = h.read_facebook_graph()
#
# # extracting ego centric networks
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_fb(original_graph, n=50, hop=2, center=True)

print("Loading Facebook 50 ego centric networks...")

with open('../Data/first_50_ego.pckl', 'rb') as f:
    ego_centric_networks, ego_nodes = pickle.load(f)

print("Analysing ego centric networks...")

num_cluster = 4
plot_cluster_settings = [{'color': 'orange', 'size': 250},
                         {'color': 'purple', 'size': 450},
                         {'color': 'gray', 'size': 650},
                         {'color': 'pink', 'size': 850}]

for i in range(0, 6):
    if nx.number_of_nodes(ego_centric_networks[i][0]) > 100:
        continue

    for j in range(0, len(ego_centric_networks[i]) - 1):
        current_snap_first_hop_nodes = ego_centric_networks[i][j].neighbors(ego_nodes[i])

        current_snap_second_hop_nodes = \
            [n for n in ego_centric_networks[i][j].nodes() if n not in current_snap_first_hop_nodes]
        current_snap_second_hop_nodes.remove(ego_nodes[i])

        next_snap_first_hop_nodes = ego_centric_networks[i][j + 1].neighbors(ego_nodes[i])

        # marking nodes which are going to form an edge with ego in the next snapshot with an X
        formed_edges_nodes_with_second_hop = \
            [n for n in next_snap_first_hop_nodes if n in current_snap_second_hop_nodes]

        # Find clusters
        gsc_model = gsc.gsc(ego_centric_networks[i][j], num_cluster)
        node_list, clusters = gsc_model.get_clusters(kmedian_max_iter=1000, num_cluster_per_node=False)
        node_list = np.array(node_list)

        # Network layout
        pos = nx.spring_layout(ego_centric_networks[i][j])

        nx.draw(ego_centric_networks[i][j], pos, node_size=1)

        # draw first hop nodes
        nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=current_snap_first_hop_nodes, node_color='y',
                               node_shape='^', node_size=50, alpha=1)

        # draw second hop nodes
        nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=current_snap_second_hop_nodes, node_color='g',
                               node_shape='s', node_size=50, alpha=1)

        # draw second hop forming nodes
        nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=formed_edges_nodes_with_second_hop,
                               node_shape='p', node_color='r', node_size=80, alpha=1)

        # draw ego node
        nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=[ego_nodes[i]], node_color='black',
                               node_shape='*', node_size=300, alpha=1)

        cluster_nodes = []
        for c in range(0, num_cluster):
            cluster_nodes = np.ndarray.tolist(node_list[np.where(clusters[:, c] == 1)[0]])
            nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=cluster_nodes,
                                   node_color=plot_cluster_settings[c]['color'],
                                   node_size=plot_cluster_settings[c]['size'], alpha=0.6 - (i / 10))

        plt.axis('off')
        plt.show()