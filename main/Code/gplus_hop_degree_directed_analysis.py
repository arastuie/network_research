import sys
import directed_graphs_helpers as dh
import networkx as nx
import numpy as np
import pickle


def gplus_run_hop_degree_directed_analysis(ego_net_file, triangle_type, overall_means, save_plot=False,
                                           plot_save_base_path=''):

    """o
    Degree based analysis in directed settings:
        background:
                * The analyzed ego-centric network consists of all the predecessors and successors of all the
                    nodes in the first hop and the ego.
                * The first hop of a network is considered to be all the successors of the ego.
                * The second hop of a network is considered to be all the successors of the nodes in the first hop.
                * A common neighbor between two nodes is the 'Z' node. This will differ based on the triangle_type.
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

    :return:
    :param ego_net_file: Name of the ego_net_file to analyze.
    :param triangle_type: This is the type of triangle that you want to test. It ranges from T01 - T09. Check out the
                Data/triangle-types-directed.png to see the types.
    :param overall_means: A dictionary consisting of 'formed_in_degree_first_hop': [],
                'not_formed_in_degree_first_hop': [], 'formed_in_degree_second_hop': [],
                'not_formed_in_degree_second_hop': [], 'formed_out_degree_first_hop': [],
                'not_formed_out_degree_first_hop': [], 'formed_out_degree_second_hop': [],
                'not_formed_out_degree_second_hop': [], to hold the overall scores.
    :param save_plot: If true, saves the plot, also a path should be passed as the next argument.
    :param plot_save_base_path: Base path to save the plot. ex: '../Plots/degree_based'
                Make sure all four different folders are in the given path. 'in_degree_first_hop, in_degree_second_hop,
                out_degree_first_hop, out_degree_second_hop'
    """
    triangle_type_func = {
        'T01': dh.get_t01_type_v_nodes,
        'T02': dh.get_t02_type_v_nodes,
        'T03': dh.get_t03_type_v_nodes,
        'T05': dh.get_t05_type_v_nodes,
        'T06': dh.get_t06_type_v_nodes
    }
    # Exit if plot should be saved, put there is no path
    if save_plot and plot_save_base_path == '':
        print(sys.stderr, "Please provide the path to which plots should be saved.")
        sys.exit(1)

    with open(ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    # Plotting related info
    ego_net_number_of_nodes = nx.number_of_nodes(ego_net)
    ego_net_number_of_edges = nx.number_of_edges(ego_net)
    tot_num_v_nodes = 0
    num_v_nodes_formed = 0
    num_v_nodes_not_formed = 0

    # return if the network has less than 30 nodes
    if ego_net_number_of_nodes < 30:
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
        temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
        ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

    snapshots_formed_z_in_degree_first_hop = []
    snapshots_formed_z_out_degree_first_hop = []
    snapshots_formed_z_in_degree_second_hop = []
    snapshots_formed_z_out_degree_second_hop = []

    snapshots_not_formed_z_in_degree_first_hop = []
    snapshots_not_formed_z_out_degree_first_hop = []
    snapshots_not_formed_z_in_degree_second_hop = []
    snapshots_not_formed_z_out_degree_second_hop = []

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes, second_hop_nodes, v_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i], ego_node)

        # Checks whether or not any edge were formed, if not skips to next snapshot
        has_any_formed = False
        for v in v_nodes:
            if ego_net_snapshots[i + 1].has_edge(ego_node, v):
                has_any_formed = True
                break

        if not has_any_formed:
            continue

        tot_num_v_nodes += len(v_nodes)

        # ANALYSIS
        formed_z_in_degree_first_hop = []
        formed_z_in_degree_second_hop = []
        formed_z_out_degree_first_hop = []
        formed_z_out_degree_second_hop = []

        not_formed_z_in_degree_first_hop = []
        not_formed_z_in_degree_second_hop = []
        not_formed_z_out_degree_first_hop = []
        not_formed_z_out_degree_second_hop = []

        for v in v_nodes:
            temp_in_degree_first_hop = []
            temp_out_degree_first_hop = []
            temp_in_degree_second_hop = []
            temp_out_degree_second_hop = []

            for z in v_nodes[v]:
                z_preds = set(ego_net_snapshots[i].predecessors(z))
                z_succs = set(ego_net_snapshots[i].successors(z))
                temp_in_degree_first_hop.append(len(z_preds.intersection(first_hop_nodes)))
                temp_in_degree_second_hop.append(len(z_preds.intersection(second_hop_nodes)))
                temp_out_degree_first_hop.append(len(z_succs.intersection(first_hop_nodes)))
                temp_out_degree_second_hop.append(len(z_succs.intersection(second_hop_nodes)))

            if ego_net_snapshots[i + 1].has_edge(ego_node, v):
                formed_z_in_degree_first_hop.append(np.mean(temp_in_degree_first_hop))
                formed_z_in_degree_second_hop.append(np.mean(temp_in_degree_second_hop))
                formed_z_out_degree_first_hop.append(np.mean(temp_out_degree_first_hop))
                formed_z_out_degree_second_hop.append(np.mean(temp_out_degree_second_hop))
                num_v_nodes_formed += 1
            else:
                not_formed_z_in_degree_first_hop.append(np.mean(temp_in_degree_first_hop))
                not_formed_z_in_degree_second_hop.append(np.mean(temp_in_degree_second_hop))
                not_formed_z_out_degree_first_hop.append(np.mean(temp_out_degree_first_hop))
                not_formed_z_out_degree_second_hop.append(np.mean(temp_out_degree_second_hop))
                num_v_nodes_not_formed += 1

        snapshots_formed_z_in_degree_first_hop.append(formed_z_in_degree_first_hop)
        snapshots_formed_z_in_degree_second_hop.append(formed_z_in_degree_second_hop)
        snapshots_formed_z_out_degree_first_hop.append(formed_z_out_degree_first_hop)
        snapshots_formed_z_out_degree_second_hop.append(formed_z_out_degree_second_hop)

        snapshots_not_formed_z_in_degree_first_hop.append(not_formed_z_in_degree_first_hop)
        snapshots_not_formed_z_in_degree_second_hop.append(not_formed_z_in_degree_second_hop)
        snapshots_not_formed_z_out_degree_first_hop.append(not_formed_z_out_degree_first_hop)
        snapshots_not_formed_z_out_degree_second_hop.append(not_formed_z_out_degree_second_hop)

    # Return if there was no V node found
    if len(snapshots_formed_z_in_degree_first_hop) == 0:
        return

    # PLOTTING
    # get the average number
    tot_num_v_nodes = tot_num_v_nodes / len(snapshots_not_formed_z_in_degree_first_hop)
    num_v_nodes_formed = num_v_nodes_formed / len(snapshots_not_formed_z_in_degree_first_hop)
    num_v_nodes_not_formed = num_v_nodes_not_formed / len(snapshots_not_formed_z_in_degree_first_hop)

    subtitle = "Ego Centric Network of Node {0} / Total Number of Nodes: {1} / Total Number of Edges: {2} \n " \
               "Avg Number of 'V' Nodes found with {3} Relation: {4:.1f} / Ones Formed: {5:.1f} / Ones Did Not " \
               "Form: {6:.1f}".format(ego_node, ego_net_number_of_nodes, ego_net_number_of_edges, triangle_type,
                                      tot_num_v_nodes, num_v_nodes_formed, num_v_nodes_not_formed)

    # in-degree first hop
    dh.plot_formed_vs_not(snapshots_formed_z_in_degree_first_hop, snapshots_not_formed_z_in_degree_first_hop,
                          xlabel="Mean In-degree of 'Z' Nodes",
                          subtitle="In-degree of 'Z' Nodes Within the First Hop \n {0}".format(subtitle),
                          overall_mean_formed=overall_means['formed_in_degree_first_hop'],
                          overall_mean_not_formed=overall_means['not_formed_in_degree_first_hop'],
                          save_plot=save_plot,
                          save_path="{0}/in_degree_first_hop/{1}.png".format(plot_save_base_path, ego_node))

    # in-degree second hop
    dh.plot_formed_vs_not(snapshots_formed_z_in_degree_second_hop, snapshots_not_formed_z_in_degree_second_hop,
                          xlabel="Mean In-degree of 'Z' Nodes",
                          subtitle="In-degree of 'Z' Nodes Within the Second Hop \n {0}".format(subtitle),
                          overall_mean_formed=overall_means['formed_in_degree_second_hop'],
                          overall_mean_not_formed=overall_means['not_formed_in_degree_second_hop'],
                          save_plot=save_plot,
                          save_path="{0}/in_degree_second_hop/{1}.png".format(plot_save_base_path, ego_node))

    # out-degree first hop
    dh.plot_formed_vs_not(snapshots_formed_z_out_degree_first_hop, snapshots_not_formed_z_out_degree_first_hop,
                          xlabel="Mean Out-degree of 'Z' Nodes",
                          subtitle="Out-degree of 'Z' Nodes Within the First Hop \n {0}".format(subtitle),
                          overall_mean_formed=overall_means['formed_out_degree_first_hop'],
                          overall_mean_not_formed=overall_means['not_formed_out_degree_first_hop'],
                          save_plot=save_plot,
                          save_path="{0}/out_degree_first_hop/{1}.png".format(plot_save_base_path, ego_node))

    # out-degree second hop
    dh.plot_formed_vs_not(snapshots_formed_z_out_degree_second_hop, snapshots_not_formed_z_out_degree_second_hop,
                          xlabel="Mean Out-degree of 'Z' Nodes",
                          subtitle="Out-degree of 'Z' Nodes Within the Second Hop \n {0}".format(subtitle),
                          overall_mean_formed=overall_means['formed_out_degree_second_hop'],
                          overall_mean_not_formed=overall_means['not_formed_out_degree_second_hop'],
                          save_plot=save_plot,
                          save_path="{0}/out_degree_second_hop/{1}.png".format(plot_save_base_path, ego_node))

    # print("OVERALL SCORES:")
    # print("In-degree First Hop:\n\tFEM:{0:.3f}\tNFEM:{1:.3f}"
    #       .format(np.mean(overall_means['formed_in_degree_first_hop']),
    #               np.mean(overall_means['not_formed_in_degree_first_hop'])))
    #
    # print("In-degree Second Hop:\n\tFEM:{0:.3f}\tNFEM:{1:.3f}"
    #       .format(np.mean(overall_means['formed_in_degree_second_hop']),
    #               np.mean(overall_means['not_formed_in_degree_second_hop'])))
    #
    # print("Out-degree First Hop:\n\tFEM:{0:.3f}\tNFEM:{1:.3f}"
    #       .format(np.mean(overall_means['formed_out_degree_first_hop']),
    #               np.mean(overall_means['not_formed_out_degree_first_hop'])))
    #
    # print("Out-degree Second Hop:\n\tFEM:{0:.3f}\tNFEM:{1:.3f}"
    #       .format(np.mean(overall_means['formed_out_degree_second_hop']),
    #               np.mean(overall_means['not_formed_out_degree_second_hop'])))

    print("\nGraph analyzed!\n")