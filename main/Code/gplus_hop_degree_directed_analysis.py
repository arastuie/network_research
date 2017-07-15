import sys
import directed_graphs_helpers as dh
import networkx as nx
import numpy as np
import pickle


def gplus_run_hop_degree_directed_analysis(ego_net_file, triangle_type, overall_means, save_plot=False,
                                           plot_save_base_path=''):

    """
    Degree based analysis in directed settings:
        background:
                * The 'directed relationships' with U, Z and, V nodes, mentioned through this documentation is referring
                    to the relationships listed out in Data/triangle-types-directed.png.
                * The analyzed ego-centric network consists of all the predecessors and successors of all the
                    nodes in the first hop and the ego.
                * The first hop of a network is considered to be all Z nodes which have the given T0X relationship with
                    the ego node.
                * The second hop of a network is considered to be all the V nodes which have the given T0X relationship
                    with any of the nodes in the first hop. Also, none of these V nodes should be in the first hop.
                * A common neighbor between the ego (U) and any of the V nodes, is the 'Z' node. This will differ based
                    on the given T0X triangle_type.
                * Notice that in this test, a new connection for the ego is a new successor not a new predecessor.

        First, read in the network and divide the given ego_net_file into 4 different snapshots, then:

        For every snapshot of the ego centric network:
            1. Find the nodes in the first and second hop.
            2. For each node V in the second hop, find all the of its common neighbors with the ego.
            3. For each common neighbor, find all of its predecessors (in-degree) and successors (out-degree), then
                check how many of them are in the first hop or the second hop, depending on the test.
            4. Check whether or not ego-node started following the V node in the next snapshot, and divide the results
                in step 3 into two groups, the ones which the ego started following and the ones that it did not.
            7. Plot a histogram comparing the result the two groups found in step 4 (Only one figure containing all the
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
        'T01': dh.get_t01_type_nodes,
        'T02': dh.get_t02_type_nodes,
        'T03': dh.get_t03_type_nodes,
        'T04': dh.get_t04_type_nodes,
        'T05': dh.get_t05_type_nodes,
        'T06': dh.get_t06_type_nodes,
        'T07': dh.get_t07_type_nodes,
        'T08': dh.get_t08_type_nodes,
        'T09': dh.get_t09_type_nodes,
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

        # Checks whether or not any edge were formed and not formed, if not skips to next snapshot
        has_any_formed = False
        has_any_not_formed = False
        for v in v_nodes:
            if ego_net_snapshots[i + 1].has_edge(ego_node, v):
                has_any_formed = True
            else:
                has_any_not_formed = True

            if has_any_formed and has_any_not_formed:
                break

        if not has_any_formed or not has_any_not_formed:
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

    plot_save_base_path += '/' + triangle_type

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
