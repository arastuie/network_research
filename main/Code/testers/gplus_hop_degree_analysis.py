import sys
import pickle
import numpy as np
import networkx as nx
import Code.helpers as h


def gplus_run_hop_degree_analysis(ego_net_file, save_plot=False, plot_save_path=''):

    """
    Degree based analysis (with the second hop):
        1. Finds all the nodes in the second hop which formed and did not form an edge with the ego node in the next
           snapshot
        2. Finds the degree of all those nodes in a given hop
        3. Plots a histogram comparing the degree of nodes in the second hop which formed an edge with the ego node vs
           the node that did not.

    :param ego_net_file: Snapshots of an ego-centric network
    :param save_plot: If true, saves the plot, also a path should be passed as the next argument.
    :param plot_save_path: Path to save the plot. ex: '../Plots/degree_based'
    """

    # Exit if plot should be saved, put there is no path
    if save_plot and plot_save_path == '':
        print(sys.stderr, "Please provide the path to which plots should be saved.")
        sys.exit(1)

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/%s' % ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    # return if the network has less than 30 nodes
    if nx.number_of_nodes(ego_net) < 30:
        return

    ego_net_snapshots = []

    # find out what snapshot the ego node first appeared in
    first_snapshot = 3
    for u, v, d in ego_net.out_edges(ego_node, data=True):
        if d['snapshot'] < first_snapshot:
            first_snapshot = d['snapshot']
            if first_snapshot == 0:
                break
    if first_snapshot != 0:
        for u, v, d in ego_net.in_edges(ego_node, data=True):
            if d['snapshot'] < first_snapshot:
                first_snapshot = d['snapshot']
                if first_snapshot == 0:
                    break

    if first_snapshot > 2:
        return

    for r in range(first_snapshot, 4):
        temp_net = nx.Graph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
        ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True))

    degree_formed_in_snapshots = []
    degree_not_formed_in_snapshots = []

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes = \
            h.get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(ego_net_snapshots[i],
                                                                                  ego_net_snapshots[i + 1],
                                                                                  ego_node, True)
        if len(formed_edges_nodes_with_second_hop) < 1:
            continue

        # <editor-fold desc="Analyze formed edges">
        # List of degrees of nodes in the second hop which formed an edge with the ego node
        degree_formed = []
        for u in formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            temp_degree_formed = []

            for c in common_neighbors:
                # first hop test
                # temp_degree_formed.append(len([n for n in ego_net_snapshots[i].neighbors(c) if n in
                #                           current_snap_first_hop_nodes]))

                # second hop test
                temp_degree_formed.append(len([n for n in ego_net_snapshots[i].neighbors(c) if n not in
                                          current_snap_first_hop_nodes]) - 1)

            degree_formed.append(np.mean(temp_degree_formed))

        # </editor-fold>
        # <editor-fold desc="Analyze not formed edges">
        # List of degrees of nodes in the second hop which did not form an edge with the ego node
        degree_not_formed = []
        for u in not_formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            temp_degree_not_formed = []

            for c in common_neighbors:
                # first hop test
                # temp_degree_not_formed.append(len([n for n in ego_net_snapshots[i].neighbors(c) if n in
                #                               current_snap_first_hop_nodes]))

                # second hop test
                temp_degree_not_formed.append(len([n for n in ego_net_snapshots[i].neighbors(c) if n not in
                                              current_snap_first_hop_nodes]) - 1)

            degree_not_formed.append(np.mean(temp_degree_not_formed))
        # </editor-fold>

        if len(degree_formed) != 0 and len(degree_not_formed) != 0:
            degree_formed_in_snapshots.append(degree_formed)
            degree_not_formed_in_snapshots.append(degree_not_formed)

    if len(degree_formed_in_snapshots) != 0:
        if save_plot:
            plot_save_path += '/%d_hop_degree.png' % ego_node

        h.plot_formed_vs_not('hop_degree', degree_formed_in_snapshots,
                             degree_not_formed_in_snapshots, plot_number=ego_node, save_plot=save_plot,
                             save_path=plot_save_path, hop_number=2)

        print("Graph analyzed!")

