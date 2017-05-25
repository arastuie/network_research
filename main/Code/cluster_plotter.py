import pickle
import sys
import helpers as h
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import GSC.generalized_spectral_clustering as gsc


def get_cluster_info(ego_network, ego_node, clusters, node_list, first_hop_nodes, second_hop_nodes, nodes_will_form,
                     next_snap_first_hop_nodes):
    info_sb = 'Ego-node ID: %d' % ego_node
    info_sb += "\n********************************************\n NUMBER OF NODES\n"
    info_sb += "\t In the first hop: %d\n" % len(first_hop_nodes)
    info_sb += "\t In the second hop: %d\n" % len(second_hop_nodes)
    info_sb += "\nAVG NUMBER OF NEIGHBORS\n"

    info_sb += "\t of nodes in the first hop: %.2f\n" % h.get_avg_num_neighbors(ego_network, first_hop_nodes)

    info_sb += "\t of nodes in the second hop: %.2f\n" % h.get_avg_num_neighbors(ego_network, second_hop_nodes)

    info_sb += "\nAverage number of CN of nodes in second hop: %.2f\n" % h.get_avg_num_cn(ego_network, ego_node,
                                                                                          second_hop_nodes)

    info_sb += "\nNumber of clusters: %d\n" % len(clusters[0])

    info_sb += "\nNUMBER OF NODES PER CLUSTER\n"

    len_of_clusters = np.sum(clusters, axis=0)
    for c in range(0, len(len_of_clusters)):
        info_sb += "\t Cluster %d: %d\n" % (c, len_of_clusters[c])

    overlap = np.round(h.get_cluster_overlap(clusters), 2)
    info_sb += "\nCLUSTERS OVERLAP - AVG: {0}\n".format((np.sum(overlap) - len(overlap[0]))
                                                 / (len(overlap[0]) ** 2 - len(overlap[0])))

    info_sb += '{0}\n'.format(overlap)
    info_sb += "\nAVG NUMBER OF CLUSTERS PER NODE\n"
    info_sb += "\tNodes in the first hop: %.2f\n" % h.get_avg_num_cluster_per_node(ego_network, clusters, node_list,
                                                                            first_hop_nodes)
    info_sb += "\tNodes in the second hop: %.2f\n" % h.get_avg_num_cluster_per_node(ego_network, clusters, node_list,
                                                                             second_hop_nodes)

    info_sb += "\nClusters containing Ego node: "
    info_sb += '{0}'.format(np.where(clusters[node_list.index(ego_node)] == 1)[0])

    info_sb += "\n\nNODE FORMING INFO\n"
    all_cn = []
    for n in range(0, len(nodes_will_form)):
        info_sb += "\tNode ID %d: \t Clusters: " % nodes_will_form[n]
        info_sb += '{0}'.format(np.where(clusters[node_list.index(nodes_will_form[n])] == 1)[0])
        info_sb += "\t CNs: "
        cn = sorted(nx.common_neighbors(ego_network, ego_node, nodes_will_form[n]))
        all_cn = np.concatenate((all_cn, cn))
        info_sb += '{0}'.format(cn)
        info_sb += "\tCN count: %d" % len(cn)
        info_sb += "\tNum neighbors: %d\n" % len(nx.neighbors(ego_network, nodes_will_form[n]))

    info_sb += "\nCOMMON NEIGHBORS INFO\n"
    unique, counts = np.unique(all_cn.astype(int), return_counts=True)
    for cn in range(0, len(unique)):
        info_sb += "\t Node ID %d: " % unique[cn]
        info_sb += "\t Count: %d" % counts[cn]
        info_sb += "\t Clusters: "
        info_sb += '{0}'.format(np.where(clusters[node_list.index(unique[cn])] == 1)[0])
        cn_neighbors = nx.neighbors(ego_network, unique[cn])
        info_sb += "\t Num neighbors(in first hop: %d, in second hop %d)\n" % (len([n for n in cn_neighbors if n in
                                              first_hop_nodes]),len([n for n in cn_neighbors if n in second_hop_nodes]))

        len([n for n in nx.neighbors(ego_network, unique[cn]) if n in first_hop_nodes])
        info_sb += "\nPercent of which ego node's new neighbors come from the second hop: {0}%\n".format(
                                   100 * len(nodes_will_form) / (len(next_snap_first_hop_nodes) - len(first_hop_nodes)))

    info_sb += "********************************************\n\n"

    file = open('../Results/txt/%d ego-net info.txt' % ego_node, 'w')
    file.write(info_sb)
    file.close()

# # reading FB graph
# original_graph = h.read_facebook_graph()
#
# # extracting ego centric networks
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_fb(original_graph, 200, "random_200_ego_nets.pckl",
#                                                                    search_type='random', hop=2, center=True)

print("Loading Facebook 200 ego centric networks...")

with open('../Data/random_200_ego_nets.pckl', 'rb') as f:
    ego_centric_networks, ego_nodes = pickle.load(f)

print("Analysing ego centric networks...")

num_cluster = 4
plot_cluster_settings = [{'color': 'orange', 'size': 250},
                         {'color': 'purple', 'size': 450},
                         {'color': 'gray', 'size': 650},
                         {'color': 'pink', 'size': 850}]
num_networks = len(ego_centric_networks)
for i in range(0, num_networks):
    if nx.number_of_nodes(ego_centric_networks[i][0]) < 25:
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

        get_cluster_info(ego_centric_networks[i][j], ego_nodes[i], clusters, node_list, current_snap_first_hop_nodes,
                         current_snap_second_hop_nodes, formed_edges_nodes_with_second_hop, next_snap_first_hop_nodes)

        # fig = plt.figure()
        # # fig = plt.figure(figsize=(30, 15), dpi=200)
        # fig.suptitle("Ego-node ID: %d - Snap number: %d" % (ego_nodes[i], j))
        # # Network layout
        # pos = nx.spring_layout(ego_centric_networks[i][j])
        #
        # nx.draw(ego_centric_networks[i][j], pos, node_size=1)
        #
        # # draw first hop nodes
        # nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=current_snap_first_hop_nodes, node_color='y',
        #                        node_shape='^', node_size=50, alpha=1)
        #
        # # draw second hop nodes
        # nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=current_snap_second_hop_nodes, node_color='g',
        #                        node_shape='s', node_size=50, alpha=1)
        #
        # # draw second hop forming nodes
        # nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=formed_edges_nodes_with_second_hop,
        #                        node_shape='p', node_color='r', node_size=80, alpha=1)
        #
        # # draw ego node
        # nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=[ego_nodes[i]], node_color='black',
        #                        node_shape='*', node_size=300, alpha=1)
        #
        # node_list = np.array(node_list)
        # cluster_nodes = []
        # for c in range(0, num_cluster):
        #     cluster_nodes = np.ndarray.tolist(node_list[np.where(clusters[:, c] == 1)[0]])
        #     nx.draw_networkx_nodes(ego_centric_networks[i][j], pos, nodelist=cluster_nodes,
        #                            node_color=plot_cluster_settings[c]['color'],
        #                            node_size=plot_cluster_settings[c]['size'], alpha=0.6 - (i / 10))
        #
        # current_fig = plt.gcf()
        # current_fig.savefig('../Results/graphs/%d ego-net graph.png' % ego_nodes[i])
        # # plt.show()

        sys.stdout.write("\rProgress: %d%%" % (100 * (1 + i) / num_networks))
        sys.stdout.flush()
        break

