import sys
import helpers as h
import networkx as nx
import numpy as np
import pickle


def gplus_run_hop_degree_directed_analysis(ego_net_file, save_plot=False, plot_save_path=''):

    """
    Degree based analysis (with the second hop in directed graph):
        background:
                * The analyzed ego-centric network consists of all the predecessors and successors of all the
                    nodes in the first hop and the ego.
                * The first hop of a network is considered to be all the the successors of the ego node.
                * The second hop of a network is considered to be all the predecessors of all the nodes in the first
                    hop, which are not in the first hop.
                * A common neighbor between two nodes is considered to be a node that is a successor of both nodes.
                * Notice that in this test, a new connection for the ego is a new successor not a new predecessor.

        First, read in the network and divide the given ego_net_file into 4 different snapshots, then:

        For every snapshot of the ego centric network:
            1. Find all the nodes in the second hop which the ego started following in the next snapshot, as well as
                those which the ego did not start to follow
            2. Find all common neighbors between the ego and all the nodes that the ego started to follow in the next
                snapshot
            3. For each common neighbor, find all of its predecessors, then check how many of them are in the first hop
                or the second hop, depending on the test
            5. Find all common neighbors between the ego and all the nodes that the ego did not start to follow in the
                next snapshot
            6. Repeat step 3 on the common neighbors found in step 5
            7. Plot a histogram comparing the result found in step 3 and step 6 (Only one figure containing all the
                plots will be plotted at the end)

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

    with open('../Data/gplus-ego/first-hop-nodes/%s' % ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    # return if the network has less than 30 nodes
    if nx.number_of_nodes(ego_net) < 30:
        return

    ego_net_snapshots = []
    one_hop_ego_net_snapshots = []

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
        temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
        ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

    degree_formed_in_snapshots = []
    degree_not_formed_in_snapshots = []

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        current_snap_first_hop_nodes = ego_net_snapshots[i].successors(ego_node)

        current_snap_second_hop_nodes = set()
        for n in current_snap_first_hop_nodes:
            current_snap_second_hop_nodes = current_snap_second_hop_nodes.union(ego_net_snapshots[i].predecessors(n))

        current_snap_second_hop_nodes = list(current_snap_second_hop_nodes - set(current_snap_first_hop_nodes))

        formed_edges_nodes_with_second_hop = \
            [n for n in ego_net_snapshots[i + 1].successors(ego_node) if n in current_snap_second_hop_nodes]

        if len(formed_edges_nodes_with_second_hop) < 1:
            continue

        not_formed_edges_nodes_with_second_hop = \
            [n for n in current_snap_second_hop_nodes if n not in formed_edges_nodes_with_second_hop]

        # <editor-fold desc="Analyze formed edges">
        # List of degrees of nodes in the second hop which formed an edge with the ego node
        degree_formed = []
        for u in formed_edges_nodes_with_second_hop:
            common_neighbors = set(ego_net_snapshots[i].successors(ego_node)).intersection(
                ego_net_snapshots[i].successors(u))

            if len(common_neighbors) == 0:
                continue

            temp_degree_formed = []

            for c in common_neighbors:
                # first hop test
                temp_degree_formed.append(len(set(ego_net_snapshots[i].predecessors(c))
                                              .intersection(current_snap_first_hop_nodes)))

                # second hop test
                # temp_degree_formed.append(len(set(ego_net_snapshots[i].predecessors(c))
                #                               .intersection(current_snap_second_hop_nodes)))

            degree_formed.append(np.mean(temp_degree_formed))

        if len(degree_formed) == 0:
            return

        # </editor-fold>
        # <editor-fold desc="Analyze not formed edges">
        # List of degrees of nodes in the second hop which did not form an edge with the ego node
        degree_not_formed = []
        for u in not_formed_edges_nodes_with_second_hop:
            common_neighbors = set(ego_net_snapshots[i].successors(ego_node)).intersection(
                ego_net_snapshots[i].successors(u))

            if len(common_neighbors) == 0:
                continue

            temp_degree_not_formed = []

            for c in common_neighbors:
                # first hop test
                temp_degree_not_formed.append(len(set(ego_net_snapshots[i].predecessors(c))
                                                  .intersection(current_snap_first_hop_nodes)))

                # second hop test
                # temp_degree_not_formed.append(len(set(ego_net_snapshots[i].predecessors(c))
                #                                   .intersection(current_snap_second_hop_nodes)))

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

