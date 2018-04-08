import sys
import numpy as np
import networkx as nx
import Code.helpers as h
import GSC.generalized_spectral_clustering as gsc


def run_percent_of_cn_belonging_to_same_cluster(ego_net_snapshots, ego_node, num_cluster, ego_net_num, save_plot=False,
                                                plot_save_path=''):

    """
    Total number of clusters per of all common neighbors analysis (with the second hop):
        1. Finds all the nodes in the second hop which formed and did not form an edge with the ego node in the next
           snapshot
        2. Finds clusters in the second hop of the snapshot
        3.
        4. Plots a histogram comparing the total number of clusters all the common neighbors of two target nodes belong
           to. Common neighbors between the ego node and the nodes in the second hop which formed an edge vs the ones
           which did not.

    :param ego_net_snapshots: Snapshots of an ego-centric network
    :param ego_node: The ego node
    :param ego_net_num: The number of the ego network, only used for the plot title
    :param num_cluster: Number of clusters to be found in the second hop of the network
    :param save_plot: If true, saves the plot, also a path should be passed as the next argument.
    :param plot_save_path: Path to save the plot. ex: '/shared/Results/EgocentricLinkPrediction/old-plots/all-plots-in-old-plots-folder/total_cluster'
    """

    # Exit if plot should be saved, put there is no path
    if save_plot and plot_save_path == '':
        print(sys.stderr, "Please provide the path to which plots should be saved.")
        sys.exit(1)

    formed_max_concentrations_in_snapshots = {}
    not_formed_max_concentrations_in_snapshots = {}

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes = \
            h.get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(ego_net_snapshots[i],
                                                                                  ego_net_snapshots[i + 1],
                                                                                  ego_node, True)

        # <editor-fold desc="Run Regularized Spectral Clustering">
        if len(current_snap_first_hop_nodes) < 50:
            continue
        gsc_model = gsc.gsc(ego_net_snapshots[i].subgraph(current_snap_first_hop_nodes), num_cluster)
        # gsc_model = gsc.gsc(ego_net_snapshots[i], num_cluster)
        node_list, clusters = gsc_model.get_clusters(kmedian_max_iter=1500, num_cluster_per_node=False)
        # </editor-fold>

        # <editor-fold desc="Analyze formed edges">
        # List of total number of clusters that all the common neighbors of a formed edge belong to
        tot_n_cluster_formed = []
        for u in formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            clusters_formed = []

            cn_count = 0
            for c in common_neighbors:
                clusters_formed.append(clusters[node_list.index(c)])
                cn_count += 1

            if cn_count == 0:
                continue

            # tot_cluster_count = np.sum(clusters_formed)
            # if tot_cluster_count == 0:
            #     continue

            max_concentration = max(np.sum(clusters_formed, axis=0) / cn_count)
            h.add_to_dic(formed_max_concentrations_in_snapshots, cn_count, max_concentration)
        # </editor-fold>

        # <editor-fold desc="Analyze not formed edges">
        # List of total number of clusters that all the common neighbors of a not formed edge belong to
        tot_n_cluster_not_formed = []
        for u in not_formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            clusters_not_formed = []

            cn_count = 0
            for c in common_neighbors:
                clusters_not_formed.append(clusters[node_list.index(c)])
                cn_count += 1

            if cn_count == 0:
                continue

            # tot_cluster_count = np.sum(clusters_not_formed)
            # if tot_cluster_count == 0:
            #     continue

            max_concentration = max(np.sum(clusters_not_formed, axis=0) / cn_count)
            h.add_to_dic(not_formed_max_concentrations_in_snapshots, cn_count, max_concentration)
        # </editor-fold>

    formed_max_concentrations_in_snapshots, not_formed_max_concentrations_in_snapshots = h.keep_common_keys(
        formed_max_concentrations_in_snapshots, not_formed_max_concentrations_in_snapshots)

    if len(formed_max_concentrations_in_snapshots) != 0:
        if save_plot:
            plot_save_path += '/ego_net_%d_cluster_concentration.png' % ego_net_num

        n_edges = nx.number_of_edges(ego_net_snapshots[len(ego_net_snapshots) - 1])
        n_nodes = nx.number_of_nodes(ego_net_snapshots[len(ego_net_snapshots) - 1])

        h.plot_formed_vs_not_dic(formed_max_concentrations_in_snapshots, not_formed_max_concentrations_in_snapshots,
                                 plot_number=ego_net_num, n_edges=n_edges, n_nodes=n_nodes, save_plot=save_plot,
                                 save_path=plot_save_path)
