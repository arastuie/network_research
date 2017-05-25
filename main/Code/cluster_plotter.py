import pickle
import helpers as h
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import GSC.generalized_spectral_clustering as gsc


def get_cluster_info(ego_network, ego_node, clusters, node_list, first_hop_nodes, second_hop_nodes, nodes_will_form,
                     next_snap_first_hop_nodes):
    print("\n********************************************")
    print("NUMBER OF NODES")
    print("\t In the first hop: %d" % len(first_hop_nodes))
    print("\t In the second hop: %d" % len(second_hop_nodes))

    print("\nAVG NUMBER OF NEIGHBORS")
    print("\t of nodes in the first hop: %.2f" % h.get_avg_num_neighbors(ego_network, first_hop_nodes))
    print("\t of nodes in the second hop: %.2f" % h.get_avg_num_neighbors(ego_network, second_hop_nodes))

    print("\nAverage number of CN of nodes in second hop: %.2f" % h.get_avg_num_cn(ego_network, ego_node,
                                                                                   second_hop_nodes))

    print("\nNumber of clusters: %d" % len(clusters[0]))

    print("\nNUMBER OF NODES PER CLUSTER")
    len_of_clusters = np.sum(clusters, axis=0)
    for c in range(0, len(len_of_clusters)):
        print("\t Cluster %d: %d" % (c, len_of_clusters[c]))

    overlap = np.round(h.get_cluster_overlap(clusters), 2)
    print("\nCLUSTERS OVERLAP - AVG: {0}".format((np.sum(overlap) - len(overlap[0]))
                                                 / (len(overlap[0]) ** 2 - len(overlap[0]))))
    print(overlap)

    print("\nAVG NUMBER OF CLUSTERS PER NODE")
    print("\tNodes in the first hop: %.2f" % h.get_avg_num_cluster_per_node(ego_network, clusters, node_list,
                                                                            first_hop_nodes))
    print("\tNodes in the second hop: %.2f" % h.get_avg_num_cluster_per_node(ego_network, clusters, node_list,
                                                                             second_hop_nodes))

    print("\nClusters containing Ego node: ", end='')
    print(np.where(clusters[node_list.index(ego_node)] == 1)[0])

    print("\nNODE FORMING INFO")
    all_cn = []
    for n in range(0, len(nodes_will_form)):
        print("\tNode ID %d: \t Clusters: " % nodes_will_form[n], end='')
        print(np.where(clusters[node_list.index(nodes_will_form[n])] == 1)[0], end='')
        print("\t CNs: ", end='')
        cn = sorted(nx.common_neighbors(ego_network, ego_node, nodes_will_form[n]))
        all_cn = np.concatenate((all_cn, cn))
        print(cn, end='')
        print("\tCN count: %d" % len(cn), end='')
        print("\tNum neighbors: %d" % len(nx.neighbors(ego_network, nodes_will_form[n])))

    print("\nCOMMON NEIGHBORS INFO")
    unique, counts = np.unique(all_cn.astype(int), return_counts=True)
    for cn in range(0, len(unique)):
        print("\t Node ID %d: " % unique[cn], end='')
        print("\t Count: %d" % counts[cn], end='')
        print("\t Clusters: ", end='')
        print(np.where(clusters[node_list.index(unique[cn])] == 1)[0], end='')
        cn_neighbors = nx.neighbors(ego_network, unique[cn])
        print("\t Num neighbors(in first hop: %d, in second hop %d)" %
              (len([n for n in cn_neighbors if n in first_hop_nodes]),
               len([n for n in cn_neighbors if n in second_hop_nodes])))

        len([n for n in nx.neighbors(ego_network, unique[cn]) if n in first_hop_nodes])
    print("\nPercent of which ego node's new neighbors come from the second hop: {0}%".format(
        100 * len(nodes_will_form) / (len(next_snap_first_hop_nodes) - len(first_hop_nodes))))
    print("********************************************\n")

# reading FB graph
original_graph = h.read_facebook_graph()

# extracting ego centric networks
ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_fb(original_graph, 200, "random_200_ego_nets.pckl",
                                                                   search_type='random', hop=2, center=True)

print("Loading Facebook 50 ego centric networks...")

with open('../Data/first_50_ego.pckl', 'rb') as f:
    ego_centric_networks, ego_nodes = pickle.load(f)

print("Analysing ego centric networks...")

num_cluster = 4
plot_cluster_settings = [{'color': 'orange', 'size': 250},
                         {'color': 'purple', 'size': 450},
                         {'color': 'gray', 'size': 650},
                         {'color': 'pink', 'size': 850}]

for i in range(0, 20):
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

        if len(formed_edges_nodes_with_second_hop) == 0:
            continue

        # Find clusters
        gsc_model = gsc.gsc(ego_centric_networks[i][j], num_cluster)
        node_list, clusters = gsc_model.get_clusters(kmedian_max_iter=1000, num_cluster_per_node=False)
        # node_list = np.array(node_list)

        print("Ego-node ID: %d - Snap number: %d" % (ego_nodes[i], j))
        get_cluster_info(ego_centric_networks[i][j], ego_nodes[i], clusters, node_list, current_snap_first_hop_nodes,
                         current_snap_second_hop_nodes, formed_edges_nodes_with_second_hop, next_snap_first_hop_nodes)

        fig = plt.figure()
        fig.suptitle("Ego-node ID: %d - Snap number: %d" % (ego_nodes[i], j))
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

        node_list = np.array(node_list)
        cluster_nodes = []
        for c in range(0, num_cluster):
            cluster_nodes = np.ndarray.tolist(node_list[np.where(clusters[:, c] == 1)[0]])
            nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=cluster_nodes,
                                   node_color=plot_cluster_settings[c]['color'],
                                   node_size=plot_cluster_settings[c]['size'], alpha=0.6 - (i / 10))

        plt.axis('off')
        plt.show()
        break
