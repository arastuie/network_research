import sys
import numpy as np
import helpers as h
import networkx as nx
import GSC.generalized_spectral_clustering as gsc


def run_num_cluster_per_cn_analysis(ego_net_snapshots, ego_node, num_cluster, ego_net_num, save_plot=False,
                                    plot_save_path=''):

    """
    Number of cluster per common neighbor analysis (with the second hop):
        1. Finds all the nodes in the second hop which formed and did not form an edge with the ego node in the next
           snapshot
        2. Finds clusters in the second hop of the snapshot
        3. Finds how many clusters every common neighbor between the ego and the nodes found in step 1 belong to
        4. Plots a histogram comparing the number of cluster each common neighbor belongs to. Common neighbors between
           the ego node and the nodes in the second hop which formed an edge vs the ones which did not.

    :param ego_net_snapshots: Snapshots of an ego-centric network
    :param ego_node: The ego node
    :param ego_net_num: The number of the ego network, only used for the plot title
    :param num_cluster: Number of clusters to be found in the second hop of the network
    :param save_plot: If true, saves the plot, also a path should be passed as the next argument.
    :param plot_save_path: Path to save the plot. ex: '../Plots/total_cluster_overall'
    """

    # Exit if plot should be saved, put there is no path
    if save_plot and plot_save_path == '':
        print(sys.stderr, "Please provide the path to which plots should be saved.")
        sys.exit(1)

    n_cluster_formed_in_snapshots = []
    n_cluster_not_formed_in_snapshots = []

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes = \
            h.get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(ego_net_snapshots[i],
                                                                                  ego_net_snapshots[i + 1],
                                                                                  ego_node, True)

        # <editor-fold desc="Run Regularized Spectral Clustering">
        if len(current_snap_first_hop_nodes) < 20:
            continue
        gsc_model = gsc.gsc(ego_net_snapshots[i].subgraph(current_snap_first_hop_nodes), num_cluster)
        node_list, clusters = gsc_model.get_clusters(kmedian_max_iter=1000, num_cluster_per_node=False)
        num_clusters = np.sum(clusters, axis=1)
        # </editor-fold>

        # <editor-fold desc="Analyze formed edges">
        # List of number of clusters that a common neighbor belongs to (common neighbors between nodes in second hop
        # which formed an edge with ego)
        n_cluster_formed = []
        for u in formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)

            for c in common_neighbors:
                n_cluster_formed.append(num_clusters[node_list.index(c)])
        # </editor-fold>

        # <editor-fold desc="Analyze not formed edges">
        # List of number of clusters that a common neighbor belongs to (common neighbors between nodes in second hop
        # which did not form an edge with ego)
        n_cluster_not_formed = []
        for u in not_formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)

            for c in common_neighbors:
                n_cluster_not_formed.append(num_clusters[node_list.index(c)])
        # </editor-fold>

        if len(n_cluster_formed) != 0 and len(n_cluster_not_formed) != 0:
            n_cluster_formed_in_snapshots.append(n_cluster_formed)
            n_cluster_not_formed_in_snapshots.append(n_cluster_not_formed)

    if len(n_cluster_formed_in_snapshots) != 0:
        if save_plot:
            plot_save_path += '/ego_net_%d_cluster_per_cn.png' % ego_net_num

        h.plot_formed_vs_not('cluster', n_cluster_formed_in_snapshots,
                             n_cluster_not_formed_in_snapshots, plot_number=ego_net_num, save_plot=save_plot,
                             save_path=plot_save_path)
