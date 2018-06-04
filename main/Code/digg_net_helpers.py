import re
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

digg_friends_file_path = '/shared/DataSets/Digg2009/raw/digg_friends.csv'


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


def read_graph_as_directed():
    print("Reading the original Digg graph...")

    original_graph = nx.DiGraph()

    with open(digg_friends_file_path, 'r') as f:
        for line in f:
            line = line.rstrip().replace('"', '').split(',')

            # only use mutual friendships and with a valid timestamp and no self loops
            if line[1] == "0" or line[2] == line[3]:
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


def divide_to_snapshots(graph, length_of_snapshots_in_days):
    orig_snapshots = []

    first_timestamp, last_timestamp = get_first_and_last_timestamps(graph)
    timestamp_duration = length_of_snapshots_in_days * 24 * 60 * 60

    num_of_snapshots = int(math.floor((last_timestamp - first_timestamp) / timestamp_duration))

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
