import re
import os
import math
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import directed_graphs_helpers as dh

digg_friends_file_path = '/shared/DataSets/Digg2009/raw/digg_friends.csv'
digg_undirected_results_file_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/digg/undirected/'
digg_results_file_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/digg/directed/'
digg_egonets_file_path = '/shared/DataSets/Digg2009/egonets'

digg_results_lp_file_path = '/shared/Results/EgocentricLinkPrediction/main/lp/digg/directed/pickle-files/'
digg_empirical_triad_ratio_result_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/digg/' \
                                              'triad-link-formed-ratio/pickle-files/'


def read_graph():
    print("Reading the original Digg graph...")

    original_graph = nx.Graph()

    with open(digg_friends_file_path, 'r') as f:
        for line in f:
            line = line.rstrip().replace('"', '').split(',')

            # only use mutual friendships and with a valid timestamp and no self loops
            if line[0] == "0" or line[1] == "0" or line[2] == line[3]:
                continue

            original_graph.add_edge(int(line[2]), int(line[3]), timestamp=int(line[1]))

    print("Digg graph in.")
    return original_graph


def get_first_and_last_timestamps(graph):
    edges = list(graph.edges(data=True))

    first_timestamp = last_timestamp = edges[0][2]['timestamp']

    for u, v, e in edges:
        if e['timestamp'] < first_timestamp:
            first_timestamp = e['timestamp']

        if e['timestamp'] > last_timestamp:
            last_timestamp = e['timestamp']

    return first_timestamp, last_timestamp


def divide_to_snapshots(graph, length_of_snapshots_in_days, directed=True):
    orig_snapshots = []

    first_timestamp, last_timestamp = get_first_and_last_timestamps(graph)
    timestamp_duration = length_of_snapshots_in_days * 24 * 60 * 60

    num_of_snapshots = int(math.floor((last_timestamp - first_timestamp) / timestamp_duration))

    if directed:
        for i in range(1, num_of_snapshots + 1):
            orig_snapshots.append(nx.DiGraph([(u, v, d) for u, v, d in graph.edges(data=True)
                                              if d['timestamp'] <= (first_timestamp + timestamp_duration * i)]))
    else:
        for i in range(1, num_of_snapshots + 1):
            orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in graph.edges(data=True)
                                            if d['timestamp'] <= (first_timestamp + timestamp_duration * i)]))

    # check if any edge is left based on the snapshot numbers
    if (last_timestamp - first_timestamp) % timestamp_duration != 0:
        orig_snapshots.append(graph)

    print("Snapshots extracted!")
    return orig_snapshots


def get_mean_ci(res, z_value):
    return z_value * np.std(res) / np.sqrt(len(res))


def extract_empirical_overall_plotter_data(all_results, z_value):
    plot_results = {
        'global-formed': [],
        'global-not-formed': [],
        'local-formed': [],
        'local-not-formed': [],
        'error': {
            'global-formed': [],
            'global-not-formed': [],
            'local-formed': [],
            'local-not-formed': []
        }
    }

    analyzed_egos = set()
    num_of_snaps = len(all_results.keys())
    for snap_index in range(num_of_snaps):
        for ego in all_results[snap_index]:
            if ego in analyzed_egos:
                continue

            temp_local_formed = []
            temp_local_not_formed = []
            temp_global_formed = []
            temp_global_not_formed = []

            # get the mean for degrees of z nodes of an ego-node in one snap
            for i in range(snap_index, num_of_snaps):

                if ego not in all_results[i]:
                    continue

                # check for a bug in the empirical analysis code
                if len(all_results[i][ego]['local_degrees_not_formed']) == 0:
                    analyzed_egos.add(ego)
                    continue

                temp_local_formed.append(
                    np.mean(all_results[i][ego]['local_degrees_formed']) / all_results[i][ego]['num_z_nodes'])
                temp_local_not_formed.append(
                    np.mean(all_results[i][ego]['local_degrees_not_formed']) / all_results[i][ego]['num_z_nodes'])

                temp_global_formed.append(
                    np.mean(all_results[i][ego]['global_degrees_formed']) / all_results[i][ego]['num_nodes'])
                temp_global_not_formed.append(
                    np.mean(all_results[i][ego]['global_degrees_not_formed']) / all_results[i][ego]['num_nodes'])

            if len(temp_local_not_formed) > 0:
                # get the mean of the degrees of an ego-node across all snaps
                plot_results['local-formed'].append(np.mean(temp_local_formed))
                plot_results['local-not-formed'].append(np.mean(temp_local_not_formed))
                plot_results['global-formed'].append(np.mean(temp_global_formed))
                plot_results['global-not-formed'].append(np.mean(temp_global_not_formed))

            analyzed_egos.add(ego)

    # getting confidence interval over all degrees of all ego-node means

    plot_results['error']['local-formed'] = get_mean_ci(plot_results['local-formed'], z_value)
    plot_results['error']['local-not-formed'] = get_mean_ci(plot_results['local-not-formed'], z_value)
    plot_results['error']['global-formed'] = get_mean_ci(plot_results['global-formed'], z_value)
    plot_results['error']['global-not-formed'] = get_mean_ci(plot_results['global-not-formed'], z_value)

    plot_results['local-formed'] = np.mean(plot_results['local-formed'])
    plot_results['local-not-formed'] = np.mean(plot_results['local-not-formed'])
    plot_results['global-formed'] = np.mean(plot_results['global-formed'])
    plot_results['global-not-formed'] = np.mean(plot_results['global-not-formed'])

    return plot_results


def extract_empirical_overall_plotter_data_test(all_results):
    plot_results = {
        'global-formed-x': [],
        'global-formed-y': [],
        'global-not-formed-x': [],
        'global-not-formed-y': [],
        'local-formed-x': [],
        'local-formed-y': [],
        'local-not-formed-x': [],
        'local-not-formed-y': [],
    }

    analyzed_egos = set()
    num_of_snaps = len(all_results.keys())
    for snap_index in range(num_of_snaps):
        for ego in all_results[snap_index]:
            if ego in analyzed_egos:
                continue

            # get the mean for degrees of z nodes of an ego-node in one snap
            for i in range(snap_index, num_of_snaps):

                if ego not in all_results[i]:
                    continue

                # check for a bug in the empirical analysis code
                if len(all_results[i][ego]['local_degrees_not_formed']) == 0:
                    analyzed_egos.add(ego)
                    continue

                for z_node in all_results[i][ego]['local_degrees_formed']:
                    plot_results['local-formed-y'].append(z_node / all_results[i][ego]['num_z_nodes'])
                    plot_results['local-formed-x'].append(i)

                for z_node in all_results[i][ego]['local_degrees_not_formed']:
                    plot_results['local-not-formed-y'].append(z_node / all_results[i][ego]['num_z_nodes'])
                    plot_results['local-not-formed-x'].append(i)

                for z_node in all_results[i][ego]['global_degrees_formed']:
                    plot_results['global-formed-y'].append(z_node / all_results[i][ego]['num_nodes'])
                    plot_results['global-formed-x'].append(i)

                for z_node in all_results[i][ego]['global_degrees_not_formed']:
                    plot_results['global-not-formed-y'].append(z_node / all_results[i][ego]['num_nodes'])
                    plot_results['global-not-formed-x'].append(i)

            analyzed_egos.add(ego)

    plt.scatter(plot_results['global-formed-x'], plot_results['global-formed-y'], c="b", alpha=0.5, marker='*',
                label="Formed")
    plt.scatter(plot_results['global-not-formed-x'], plot_results['global-not-formed-y'], c="r", alpha=0.5, marker='o',
                label="Not Formed")
    plt.xlabel("Snapshot Index")
    plt.ylabel("Global Degree")
    plt.legend(loc=2)
    plt.show()
    plt.clf()

    plt.scatter(plot_results['local-formed-x'], plot_results['local-formed-y'], c="b", alpha=0.5, marker='*',
                label="Formed")
    plt.scatter(plot_results['local-not-formed-x'], plot_results['local-not-formed-y'], c="r", alpha=0.5, marker='o',
                label="Not Formed")
    plt.xlabel("Snapshot Index")
    plt.ylabel("Local Degree")
    plt.legend(loc=2)
    plt.show()


#################################
####         Directed        ####
#################################

def read_graph_as_directed():
    print("Reading the original directed Digg graph...")

    original_graph = nx.DiGraph()

    with open(digg_friends_file_path, 'r') as f:
        for line in f:
            line = line.rstrip().replace('"', '').split(',')

            # use only valid timestamp and no self loops
            if line[1] == "0" or line[2] == line[3]:
                continue

            # Ignore mutual friendship flag, if the other user is active, the friendship shows up again in the dataset.
            # Else, the person will have an out-degree of zero and will be deleted.
            original_graph.add_edge(int(line[2]), int(line[3]), timestamp=int(line[1]))

    nodes = list(original_graph.nodes)
    for node in nodes:
        if original_graph.out_degree(node) == 0:
            original_graph.remove_node(node)

    print("Digg graph in.")
    return original_graph


def extract_all_egonets(snapshot_duration_in_days=90):
    orig_snaps = divide_to_snapshots(read_graph_as_directed(), snapshot_duration_in_days)

    extracted_nodes = set()

    n = orig_snaps[len(orig_snaps) - 1].number_of_nodes()

    print("Start extracting {} egonets...".format(n))

    snaps_nodes = []
    for snap_index in range(len(orig_snaps) - 1):
        nodes_in_snap = set(orig_snaps[snap_index].nodes())
        nodes_to_extract = nodes_in_snap - extracted_nodes
        snaps_nodes.append(nodes_to_extract)
        extracted_nodes = extracted_nodes.union(nodes_to_extract)

    Parallel(n_jobs=15)(delayed(extract_egonets_from)(orig_snaps, node_list, snap_index) for snap_index, node_list in
                        enumerate(snaps_nodes))


def extract_egonets_from(orig_snaps, nodes_to_extract, snap_index):
    for ego in nodes_to_extract:
        ego_snaps = []
        for i in range(snap_index, len(orig_snaps)):
            ego_snaps.append(nx.ego_graph(orig_snaps[i], ego, radius=2, center=True, undirected=True))

        with open('{}/{}.pckl'.format(digg_egonets_file_path, ego), 'wb') as f:
            pickle.dump([ego, ego_snaps], f, protocol=-1)

        print('{} egonet extracted!'.format(ego))


def run_local_degree_empirical_analysis(ego_net_file):
    # return if the egonet is on the analyzed list
    if os.path.isfile(digg_results_file_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if os.path.isfile(digg_results_file_path + 'skipped_egonets/' + ego_net_file):
        return

    # return if the egonet is on the currently being analyzed list
    if os.path.isfile(digg_results_file_path + 'temp-analyses-start/' + ego_net_file):
        return

    with open(digg_results_file_path + 'temp-analyses-start/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

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

    with open(digg_egonets_file_path + '/' + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
    if nx.number_of_nodes(ego_net_snapshots[0]) > 100000:
        with open(digg_results_file_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)

        return

    for triangle_type in triangle_type_func.keys():
        local_snapshots_formed_z_in_degree = []
        local_snapshots_formed_z_out_degree = []
        local_snapshots_not_formed_z_in_degree = []
        local_snapshots_not_formed_z_out_degree = []

        global_snapshots_formed_z_in_degree = []
        global_snapshots_formed_z_out_degree = []
        global_snapshots_not_formed_z_in_degree = []
        global_snapshots_not_formed_z_out_degree = []

        # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
        for i in range(len(ego_net_snapshots) - 1):
            first_hop_nodes, second_hop_nodes, v_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i],
                                                                                           ego_node)

            len_first_hop = len(first_hop_nodes)
            tot_num_nodes = nx.number_of_nodes(ego_net_snapshots[i])

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

            # ANALYSIS
            local_formed_z_in_degree = []
            local_formed_z_out_degree = []
            local_not_formed_z_in_degree = []
            local_not_formed_z_out_degree = []

            global_formed_z_in_degree = []
            global_formed_z_out_degree = []
            global_not_formed_z_in_degree = []
            global_not_formed_z_out_degree = []

            for v in v_nodes:
                local_temp_in_degree = []
                local_temp_out_degree = []

                global_temp_in_degree = []
                global_temp_out_degree = []

                for z in v_nodes[v]:
                    z_preds = set(ego_net_snapshots[i].predecessors(z))
                    z_succs = set(ego_net_snapshots[i].successors(z))

                    local_temp_in_degree.append(len(z_preds.intersection(first_hop_nodes)))
                    local_temp_out_degree.append(len(z_succs.intersection(first_hop_nodes)))

                    global_temp_in_degree.append(len(z_preds))
                    global_temp_out_degree.append(len(z_succs))

                if ego_net_snapshots[i + 1].has_edge(ego_node, v):
                    local_formed_z_in_degree.append(np.mean(local_temp_in_degree))
                    local_formed_z_out_degree.append(np.mean(local_temp_out_degree))

                    global_formed_z_in_degree.append(np.mean(global_temp_in_degree))
                    global_formed_z_out_degree.append(np.mean(global_temp_out_degree))
                else:
                    local_not_formed_z_in_degree.append(np.mean(local_temp_in_degree))
                    local_not_formed_z_out_degree.append(np.mean(local_temp_out_degree))

                    global_not_formed_z_in_degree.append(np.mean(global_temp_in_degree))
                    global_not_formed_z_out_degree.append(np.mean(global_temp_out_degree))

            # normalizing by the number of nodes in the first hop
            local_snapshots_formed_z_in_degree.append(np.mean(local_formed_z_in_degree) / len_first_hop)
            local_snapshots_formed_z_out_degree.append(np.mean(local_formed_z_out_degree) / len_first_hop)
            local_snapshots_not_formed_z_in_degree.append(np.mean(local_not_formed_z_in_degree) / len_first_hop)
            local_snapshots_not_formed_z_out_degree.append(np.mean(local_not_formed_z_out_degree) / len_first_hop)

            # normalizing by the number of nodes in the entire snapshot
            global_snapshots_formed_z_in_degree.append(np.mean(global_formed_z_in_degree) / tot_num_nodes)
            global_snapshots_formed_z_out_degree.append(np.mean(global_formed_z_out_degree) / tot_num_nodes)
            global_snapshots_not_formed_z_in_degree.append(np.mean(global_not_formed_z_in_degree) / tot_num_nodes)
            global_snapshots_not_formed_z_out_degree.append(np.mean(global_not_formed_z_out_degree) / tot_num_nodes)

        # Return if there was no V node found
        if len(local_snapshots_formed_z_in_degree) == 0:
            continue

        with open(digg_results_file_path + triangle_type + '/' + ego_net_file, 'wb') as f:
            pickle.dump([np.mean(local_snapshots_formed_z_in_degree),
                         np.mean(global_snapshots_formed_z_in_degree),
                         np.mean(local_snapshots_formed_z_out_degree),
                         np.mean(global_snapshots_formed_z_out_degree),
                         np.mean(local_snapshots_not_formed_z_in_degree),
                         np.mean(global_snapshots_not_formed_z_in_degree),
                         np.mean(local_snapshots_not_formed_z_out_degree),
                         np.mean(global_snapshots_not_formed_z_out_degree)], f, protocol=-1)

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(digg_results_file_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    # remove temp analyze file
    os.remove(digg_results_file_path + 'temp-analyses-start/' + ego_net_file)

    print("Analyzed ego net {0}".format(ego_net_file))
