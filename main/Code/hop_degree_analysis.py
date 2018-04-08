import sys
import numpy as np
import networkx as nx
import Code.helpers as h


def get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(current_snapshot, next_snapshot, ego_node):
    current_snap_first_hop_nodes = set(current_snapshot.neighbors(ego_node))

    current_snap_second_hop_nodes = set(current_snapshot.nodes()) - current_snap_first_hop_nodes
    current_snap_second_hop_nodes.remove(ego_node)

    next_snap_first_hop_nodes = set(next_snapshot.neighbors(ego_node))

    formed_edges_nodes_with_second_hop = next_snap_first_hop_nodes.intersection(current_snap_second_hop_nodes)

    not_formed_edges_nodes_with_second_hop = current_snap_second_hop_nodes - formed_edges_nodes_with_second_hop

    return formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes


def run_hop_local_degree_analysis(ego_net_snapshots, ego_node, snap_range, ego_net_num, save_plot=False, plot_save_path=''):

    """
    Degree based analysis (with the second hop):
        1. Finds all the nodes in the second hop which formed and did not form an edge with the ego node in the next
           snapshot
        2. Finds the degree of all those nodes in a given hop
        3. Plots a histogram comparing the degree of nodes in the second hop which formed an edge with the ego node vs
           the node that did not.

    :param ego_net_snapshots: Snapshots of an ego-centric network
    :param ego_node: The ego node
    :param ego_net_num: The number of the ego network, only used for the plot title
    :param save_plot: If true, saves the plot, also a path should be passed as the next argument.
    :param plot_save_path: Path to save the plot. ex: '/shared/Results/EgocentricLinkPrediction/old-plots/all-plots-in-old-plots-folder/degree_based'
    """

    # Exit if plot should be saved, put there is no path
    if save_plot and plot_save_path == '':
        print(sys.stderr, "Please provide the path to which plots should be saved.")
        sys.exit(1)

    degree_formed_in_snapshots = []
    degree_not_formed_in_snapshots = []
    num_first_hop_nodes_in_snapshots = []

    for i in snap_range:
        formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes = \
            get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(ego_net_snapshots[i],
                                                                                ego_net_snapshots[i + 1], ego_node)

        num_first_hop_nodes = nx.degree(ego_net_snapshots[i], ego_node)

        if len(formed_edges_nodes_with_second_hop) == 0 or num_first_hop_nodes == 0:
            continue

        # <editor-fold desc="Analyze formed edges">
        # List of degrees of nodes in the second hop which formed an edge with the ego node
        degree_formed = []
        for u in formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            temp_degree_formed = []

            # u_neighbors = set(nx.neighbors(ego_net_snapshots[i], u))

            for c in common_neighbors:
                # first hop test
                temp_degree_formed.append(len([n for n in ego_net_snapshots[i].neighbors(c) if n in
                                          current_snap_first_hop_nodes]))

                # second hop test
                # temp_degree_formed.append(len([n for n in ego_net_snapshots[i].neighbors(c) if n not in
                #                           current_snap_first_hop_nodes]) - 1)

                # Overlap with second hop node's neighbor test
                # temp_degree_formed.append(len(u_neighbors.intersection(ego_net_snapshots[i].neighbors(c))))

            degree_formed.append(np.mean(temp_degree_formed))
        # </editor-fold>
        # <editor-fold desc="Analyze not formed edges">
        # List of degrees of nodes in the second hop which did not form an edge with the ego node
        degree_not_formed = []
        for u in not_formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            temp_degree_not_formed = []

            # u_neighbors = set(nx.neighbors(ego_net_snapshots[i], u))

            for c in common_neighbors:
                # first hop test
                temp_degree_not_formed.append(len([n for n in ego_net_snapshots[i].neighbors(c) if n in
                                              current_snap_first_hop_nodes]))

                # second hop test
                # temp_degree_not_formed.append(len([n for n in ego_net_snapshots[i].neighbors(c) if n not in
                #                               current_snap_first_hop_nodes]) - 1)

                # Overlap with second hop node's neighbor test
                # temp_degree_not_formed.append(len(u_neighbors.intersection(ego_net_snapshots[i].neighbors(c))))

            degree_not_formed.append(np.mean(temp_degree_not_formed))
        # </editor-fold>

        if len(degree_formed) != 0 and len(degree_not_formed) != 0:
            degree_formed_in_snapshots.append(degree_formed)
            degree_not_formed_in_snapshots.append(degree_not_formed)
            num_first_hop_nodes_in_snapshots.append(num_first_hop_nodes)

    if len(degree_formed_in_snapshots) != 0:
        if save_plot:
            plot_save_path += '/ego_net_%d_hop_degree.png' % ego_net_num

        mfem, mnfem = h.plot_formed_vs_not_local_degree(degree_formed_in_snapshots, degree_not_formed_in_snapshots,
                                                        num_first_hop_nodes_in_snapshots, ego_net_num,
                                                        save_plot=save_plot, save_path=plot_save_path)

        print("Graph analyzed! {0}".format(ego_net_num))
        return mfem, mnfem

    print("Graph analyzed! {0}".format(ego_net_num))
    return 0, 0