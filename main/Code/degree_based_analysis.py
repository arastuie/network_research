import sys
import helpers as h
import networkx as nx


def run_degree_based_analysis(ego_net_snapshots, ego_node, ego_net_num, save_plot=False, plot_save_path=''):

    """
    Degree based analysis (with the second hop):
        1. Finds all the nodes in the second hop which formed and did not form an edge with the ego node in the next
           snapshot
        2. Finds the degree of all those nodes
        3. Plots a histogram comparing the degree of nodes in the second hop which formed an edge with the ego node vs
           the node that did not.

    :param ego_net_snapshots: Snapshots of an ego-centric network
    :param ego_node: The ego node
    :param ego_net_num: The number of the ego network, only used for the plot title
    :param save_plot: If true, saves the plot, also a path should be passed as the next argument.
    :param plot_save_path: Path to save the plot. ex: '../Plots/degree_based'
    """

    # Exit if plot should be saved, put there is no path
    if save_plot and plot_save_path == '':
        print(sys.stderr, "Please provide the path to which plots should be saved.")
        sys.exit(1)

    degree_formed_in_snapshots = []
    degree_not_formed_in_snapshots = []

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes = \
            h.get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(ego_net_snapshots[i],
                                                                                  ego_net_snapshots[i + 1],
                                                                                  ego_node, True)

        # <editor-fold desc="Analyze formed edges">
        # List of degrees of nodes in the second hop which formed an edge with the ego node
        degree_formed = []
        for u in formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)

            for c in common_neighbors:
                degree_formed.append(len(ego_net_snapshots[i].neighbors(c)))
        # </editor-fold>

        # <editor-fold desc="Analyze not formed edges">
        # List of degrees of nodes in the second hop which did not form an edge with the ego node
        degree_not_formed = []
        for u in not_formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)

            for c in common_neighbors:
                degree_not_formed.append(len(ego_net_snapshots[i].neighbors(c)))
        # </editor-fold>

        if len(degree_formed) != 0 and len(degree_not_formed) != 0:
            degree_formed_in_snapshots.append(degree_formed)
            degree_not_formed_in_snapshots.append(degree_not_formed)

    if len(degree_formed_in_snapshots) != 0:
        if save_plot:
            plot_save_path += '/ego_net_%d_degree.png' % ego_net_num

        h.plot_formed_vs_not('degree', degree_formed_in_snapshots,
                             degree_not_formed_in_snapshots, plot_number=ego_net_num, save_plot=save_plot,
                             save_path=plot_save_path)
