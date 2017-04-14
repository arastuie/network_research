import pickle
import numpy as np
import helpers as h
import networkx as nx
import generalized_spectral_clustering as gsc


# reading FB graph
# original_graph = h.read_facebook_graph()

# extracting ego centric networks
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_fb(original_graph, n=50, hop=2, center=True)

print("Loading Facebook largest 50 ego centric networks...")

with open('../Data/biggest_50_ego.pckl', 'rb') as f:
    ego_centric_networks, ego_nodes = pickle.load(f)

print("Analysing ego centric networks...")

num_cluster = 10

for o in range(len(ego_centric_networks)):
    print(o)
    ego_net_snapshots = ego_centric_networks[o]
    ego_node = ego_nodes[o]

    n_edges = nx.number_of_edges(ego_net_snapshots[len(ego_net_snapshots) - 1])
    n_nodes = nx.number_of_nodes(ego_net_snapshots[len(ego_net_snapshots) - 1])

    # skip the network if the last snapshot has less than 20 edges
    if n_edges < 20:
        print("Network skipped. \n")
        continue

    degree_formed_in_snapshots = []
    degree_not_formed_in_snapshots = []

    n_cluster_formed_in_snapshots = []
    n_cluster_not_formed_in_snapshots = []

    tot_n_cluster_formed_in_snapshots = []
    tot_n_cluster_not_formed_in_snapshots = []

    tot_n_cluster_formed_overall = {}
    tot_n_cluster_not_formed_overall = {}

    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes_t0 = ego_net_snapshots[i].neighbors(ego_node)

        # continue if there are less nodes in the network than the number of clusters
        if len(first_hop_nodes_t0) <= 20:
            continue

        second_hop_nodes_t0 = [n for n in ego_net_snapshots[i].nodes() if n not in first_hop_nodes_t0]
        second_hop_nodes_t0.remove(ego_node)

        first_hop_nodes_t1 = ego_net_snapshots[i + 1].neighbors(ego_node)
        formed_edges_nodes_with_second_hop = [n for n in first_hop_nodes_t1 if n in second_hop_nodes_t0]
        not_formed_edges_nodes_with_second_hop = [n for n in second_hop_nodes_t0 if n not in formed_edges_nodes_with_second_hop]

        # run regularized spectral clustering
        # gsc_model = gsc.gsc(ego_net_snapshots[i].subgraph(first_hop_nodes_t0), num_cluster)
        # node_list, clusters = gsc_model.get_clusters(kmedian_max_iter=1000, num_cluster_per_node=False)
        # num_clusters = np.sum(clusters, axis=1)
        # num_clusters = h.get_cluster_coefficient(clusters)

        # analyse formed edges
        degree_formed = []
        # n_cluster_formed = []
        # tot_n_cluster_formed = []
        for u in formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            # clusters_formed = []
            # n_cn = 0

            for c in common_neighbors:
                degree_formed.append(len(ego_net_snapshots[i].neighbors(c)))
                # n_cluster_formed.append(num_clusters[node_list.index(c)])
                # clusters_formed.append(clusters[node_list.index(c)])
                # n_cn += 1

            # clusters_formed = np.sum(clusters_formed, axis=0)
            # clusters_formed[clusters_formed > 0] = 1
            # clusters_formed = np.sum(clusters_formed)
            # tot_n_cluster_formed.append(clusters_formed)
            # h.add_to_dic(tot_n_cluster_formed_overall, n_cn, clusters_formed)

        # analyse not formed edges
        degree_not_formed = []
        # n_cluster_not_formed = []
        # tot_n_cluster_not_formed = []
        for u in not_formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            # clusters_not_formed = []
            # n_cn = 0

            for c in common_neighbors:
                degree_not_formed.append(len(ego_net_snapshots[i].neighbors(c)))
                # n_cluster_not_formed.append(num_clusters[node_list.index(c)])
                # clusters_not_formed.append(clusters[node_list.index(c)])
                # n_cn += 1

            # clusters_not_formed = np.sum(clusters_not_formed, axis=0)
            # clusters_not_formed[clusters_not_formed > 0] = 1
            # clusters_not_formed = np.sum(clusters_not_formed)
            # tot_n_cluster_not_formed.append(clusters_not_formed)
            # h.add_to_dic(tot_n_cluster_not_formed_overall, n_cn, clusters_not_formed)

        if len(degree_formed) != 0 and len(degree_not_formed) != 0:
            degree_formed_in_snapshots.append(degree_formed)
            degree_not_formed_in_snapshots.append(degree_not_formed)

        # if len(n_cluster_formed) != 0 and len(n_cluster_not_formed) != 0:
        #     n_cluster_formed_in_snapshots.append(n_cluster_formed)
        #     n_cluster_not_formed_in_snapshots.append(n_cluster_not_formed)

        # if len(tot_n_cluster_formed) != 0 and len(tot_n_cluster_not_formed) != 0:
        #     tot_n_cluster_formed_in_snapshots.append(tot_n_cluster_formed)
        #     tot_n_cluster_not_formed_in_snapshots.append(tot_n_cluster_not_formed)

    if len(degree_formed_in_snapshots) != 0:
        h.plot_formed_vs_not('degree', degree_formed_in_snapshots,
                             degree_not_formed_in_snapshots, plot_number=o, save_plot=False,
                             save_path='../Plots/degree_based/ego_net_%d_degree.png' % o)

    # if len(n_cluster_formed_in_snapshots) != 0:
    #     h.plot_formed_vs_not('cluster', n_cluster_formed_in_snapshots,
    #                          n_cluster_not_formed_in_snapshots, plot_number=o, save_plot=False,
    #                          save_path='../Plots/cluster_per_node/ego_net_%d_cluster.png' % o)

    # if len(tot_n_cluster_formed_in_snapshots) != 0:
    #     h.plot_formed_vs_not('tot_cluster', tot_n_cluster_formed_in_snapshots,
    #                          tot_n_cluster_not_formed_in_snapshots, plot_number=o, save_plot=True,
    #                          save_path='../Plots/total_cluster/ego_net_%d_tot_cluster.png' % o)

    # tot_n_cluster_formed_overall, tot_n_cluster_not_formed_overall = h.keep_common_keys(
    #     tot_n_cluster_formed_overall, tot_n_cluster_not_formed_overall)
    #
    # if len(tot_n_cluster_formed_overall) != 0:
    #     h.plot_formed_vs_not_dic(tot_n_cluster_formed_overall, tot_n_cluster_not_formed_overall,
    #                              plot_number=o, n_edges=n_edges, n_nodes=n_nodes, save_plot=True,
    #                              save_path='../Plots/total_cluster_overall/ego_net_%d_tot_cluster.png' % o)



