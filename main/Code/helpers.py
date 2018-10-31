import re
import sys
import math
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def plot_formed_vs_not(plot_type, formed, not_formed, plot_number, save_plot=False, save_path='', hop_number=1):
    fig = None

    if save_plot:
        fig = plt.figure(figsize=(15, 8), dpi=100)
    else:
        fig = plt.figure()

    n_row = math.ceil(len(formed) / 5)
    n_col = math.ceil(len(not_formed) / n_row)
    for i in range(len(formed)):
        formed_mean = np.mean(formed[i])
        not_formed_mean = np.mean(not_formed[i])
        p = fig.add_subplot(n_row, n_col, i + 1)

        p.hist(formed[i], color='r', alpha=0.8, weights=np.zeros_like(formed[i]) + 1. / len(formed[i]),
               label="FEM: {0:.2f}".format(formed_mean))

        p.hist(not_formed[i], color='b', alpha=0.5, weights=np.zeros_like(not_formed[i]) + 1. / len(not_formed[i]),
               label="NFEM: {0:.2f}".format(not_formed_mean))

        p.legend(loc='upper right')
        plt.ylabel('Relative Frequency')

        if plot_type == "degree":
            plt.xlabel('Degree of Common Neighbor')
            plt.suptitle('Degree of Common Neighbors \n Ego Centric Network of Node %d' % plot_number)
        elif plot_type == 'cluster':
            plt.xlabel('(*)')
            plt.suptitle('Number of Clusters Each Common Neighbor Belongs to (*) \n'
                         'Ego Centric Network of Node %d' % plot_number)
        elif plot_type == 'tot_cluster':
            plt.xlabel('(*)')
            plt.suptitle('Total Number of Clusters All Common Neighbors Belong to (*) \n'
                         'Ego Centric Network of Node %d' % plot_number)
        elif plot_type == 'hop_degree':
            plt.xlabel('Degree of Common Neighbor')
            if hop_number == 1:
                plt.suptitle('Degree of Common Neighbors within the first hop \n Ego Centric Network of Node %d' % plot_number)
            elif hop_number == 2:
                plt.suptitle('Degree of Common Neighbors outside of the first hop \n Ego Centric Network of Node %d' % plot_number)

    if not save_plot:
        plt.show()
    else:
        current_fig = plt.gcf()
        current_fig.savefig(save_path)

    plt.close(fig)


def plot_formed_vs_not_local_degree(formed, not_formed, num_first_hop_nodes, plot_number, save_plot=False, save_path=''):
    fig = None

    if save_plot:
        fig = plt.figure(figsize=(15, 8), dpi=100)
    else:
        fig = plt.figure()

    n_row = math.ceil(len(formed) / 5)
    n_col = math.ceil(len(not_formed) / n_row)

    formed_means = []
    not_formed_means = []
    for i in range(len(formed)):
        formed_mean = np.mean(formed[i])
        # normalize over the number of nodes in the first hop
        formed_means.append(formed_mean / num_first_hop_nodes[i])

        not_formed_mean = np.mean(not_formed[i])
        # normalize over the number of nodes in the first hop
        not_formed_means.append(not_formed_mean / num_first_hop_nodes[i])

        p = fig.add_subplot(n_row, n_col, i + 1)

        p.hist(formed[i], color='r', alpha=0.8, weights=np.zeros_like(formed[i]) + 1. / len(formed[i]),
               label="FEM: {0:.4f}".format(formed_mean))

        p.hist(not_formed[i], color='b', alpha=0.5, weights=np.zeros_like(not_formed[i]) + 1. / len(not_formed[i]),
               label="NFEM: {0:.4f}".format(not_formed_mean))

        p.legend(loc='upper right')
        plt.ylabel('Relative Frequency')
        plt.xlabel('Global Degree of Common Neighbors')
        plt.suptitle('Global Degree of Common Neighbors \n Ego Centric Network of Node %d' % plot_number)

    if not save_plot:
        plt.show()
    else:
        current_fig = plt.gcf()
        current_fig.savefig(save_path)

    plt.close(fig)

    mfem = np.mean(formed_means)
    mnfem = np.mean(not_formed_means)

    return mfem, mnfem


def plot_formed_vs_not_dic(formed, not_formed, plot_number, n_edges, n_nodes, save_plot=False, save_path=''):
    fig = plt.figure()

    if save_plot:
        fig = plt.figure(figsize=(30, 15), dpi=200)

    n_row = math.ceil(len(formed) / 5)
    n_col = math.ceil(len(not_formed) / n_row)
    count = 1
    for key in formed:
        formed_mean = np.mean(formed.get(key))
        not_formed_mean = np.mean(not_formed.get(key))

        p = fig.add_subplot(n_row, n_col, count)

        p.hist(formed.get(key), color='r', alpha=0.8,
               weights=np.zeros_like(formed.get(key)) + 1. / len(formed.get(key)),
               label="FEM: {0:.2f}".format(formed_mean))

        p.hist(not_formed.get(key), color='b', alpha=0.5,
               weights=np.zeros_like(not_formed.get(key)) + 1. / len(not_formed.get(key)),
               label="NFEM: {0:.2f}".format(not_formed_mean))

        p.legend(loc='upper right')
        plt.ylabel('Relative Frequency')
        plt.xlabel('(*)')
        plt.title("NCN: %d" % key)
        plt.suptitle('Percent of which CNs belong to the same cluster (*) \n'
                     'Ego Centric Network of Node %d. n_edges: %d | n_nodes: %d | edge to node ratio: %d'
                     % (plot_number, n_edges, n_nodes, n_edges / n_nodes))
        count += 1

    if not save_plot:
        plt.show()
    else:
        current_fig = plt.gcf()
        current_fig.savefig(save_path)
    #fig.clf()
    #plt.clf()


def add_to_dic(dic, key, value):
    if dic.get(key) is not None:
        dic.get(key).append(value)
    else:
        dic[key] = [value]


def keep_common_keys(dic_1, dic_2):
    uncommon = []
    for key in dic_1:
        if dic_2.get(key) is None:
            uncommon.append(key)

    for e in uncommon:
        dic_1.pop(e)

    uncommon = []
    for key in dic_2:
        if dic_1.get(key) is None:
            uncommon.append(key)

    for e in uncommon:
        dic_2.pop(e)

    return dic_1, dic_2


# Returns n x n matrix with the percent of overlap of each cluster over another
# entry ij shows what percent of nodes in i are also in j
def get_cluster_overlap(hard_cluster):
    k = np.shape(hard_cluster)[1]
    cluster_overlap = np.ones((k, k))
    cluster_size = np.sum(hard_cluster, axis=0)
    # cluster_size_avg = np.mean(cluster_size)
    for i in range(k):
        # cluster_overlap[i, i] = cluster_size_avg / cluster_size[i]
        for j in range(i + 1, k):
            n_overlap = np.sum(np.where(np.sum(hard_cluster[:, [i, j]], axis=1) == 2, 1, 0))
            cluster_overlap[i, j] = n_overlap / cluster_size[i]
            cluster_overlap[j, i] = n_overlap / cluster_size[j]

    return cluster_overlap


# Can be improved
def get_cluster_coefficient(hard_cluster):
    cluster_overlap = get_cluster_overlap(hard_cluster)
    cluster_count = np.sum(hard_cluster, axis=1)
    cluster_coefficient = np.ones(np.shape(cluster_count))
    for i in range(np.shape(hard_cluster)[0]):
        if cluster_count[i] <= 1:
            cluster_coefficient[i] = cluster_count[i]
            continue
        s = 0
        overlap_index = np.where(hard_cluster[i] == 1)[0]
        for m in overlap_index:
            for n in overlap_index:
                s += cluster_overlap[m, n]
        # if s == 0:
        #     continue
        cluster_coefficient[i] = cluster_count[i] * len(overlap_index) ** 2 / s

    return cluster_coefficient


def get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(current_snapshot, next_snapshot, ego_node,
                                                                        return_first_hop_nodes=False):
    """
    Divides all the nodes in the second hop of the current network into two groups:
        1. All the nodes which formed an edge with the ego node in the next snapshot
        2. All the ndoes which did not form an edge with the ego node in the next snapshot

    :param current_snapshot: Snapshot of the current (base) network
    :param next_snapshot: Snapshot of the next network
    :param ego_node: The ego node of the whole ego centric network
    :param return_first_hop_nodes: if true, also returns all the nodes in the first hop of the current snapshot
    """
    current_snap_first_hop_nodes = current_snapshot.neighbors(ego_node)

    current_snap_second_hop_nodes = \
        [n for n in current_snapshot.nodes() if n not in current_snap_first_hop_nodes]
    current_snap_second_hop_nodes.remove(ego_node)

    next_snap_first_hop_nodes = next_snapshot.neighbors(ego_node)

    formed_edges_nodes_with_second_hop = \
        [n for n in next_snap_first_hop_nodes if n in current_snap_second_hop_nodes]

    not_formed_edges_nodes_with_second_hop = \
        [n for n in current_snap_second_hop_nodes if n not in formed_edges_nodes_with_second_hop]

    if return_first_hop_nodes:
        return formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes
    else:
        return formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop


def get_avg_num_neighbors(network, nodes=[]):
    if len(nodes) == 0:
        nodes = network.nodes()

    if len(nodes) == 0:
        return 0

    avg = len(list(nx.neighbors(network, nodes[0])))
    count = 1
    for n in range(1, len(nodes)):
        avg = (avg * count + len(list(nx.neighbors(network, nodes[n])))) / (count + 1)
        count += 1

    return avg


# returns average number of common neighbors between ego-node and the node list
def get_avg_num_cn(ego_net, ego_node, nodes):
    if len(nodes) == 0:
        return 0

    avg = len(sorted(nx.common_neighbors(ego_net, ego_node, nodes[0])))
    count = 1
    for n in range(1, len(nodes)):
        avg = (avg * count + len(sorted(nx.common_neighbors(ego_net, ego_node, nodes[n])))) / (count + 1)
        count += 1

    return avg


def get_avg_num_cluster_per_node(network, clusters, cluster_node_list, nodes=[]):
    if len(nodes) == 0:
        nodes = network.nodes()

    if len(nodes) == 0:
        return 0

    num_clusters = np.sum(clusters, axis=1)
    avg = num_clusters[cluster_node_list.index(nodes[0])]
    count = 1
    for n in range(1, len(nodes)):
        avg = (avg * count + num_clusters[cluster_node_list.index(nodes[n])]) / (count + 1)
        count += 1

    return avg


# ECDF plotting
def get_ecdf_bands(data, alpha):
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = np.sort(data)

    n = len(data)
    if n == 0:
        return

    epsilon = math.sqrt((math.log(2 / alpha)) / (2 * n))

    edf = np.zeros(np.shape(data))
    for i in range(0, n):
        edf[i] = sum(np.where(data <= data[i], 1, 0)) / n

    lower_band = edf - epsilon
    lower_band[np.where(lower_band < 0)[0]] = 0

    upper_band = edf + epsilon
    upper_band[np.where(upper_band > 1)[0]] = 1

    return lower_band, upper_band


def add_ecdf_with_band_plot(data, lb, ub, label, color):
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = np.sort(data)

    plt.step(data, np.arange(1, len(data) + 1) / np.float(len(data)), alpha=0.9, color=color,
             label='{0}: {1:.4f}'.format(label, np.mean(data)), lw=2)

    plt.plot(data, lb, '--', color=color, alpha=0.4)
    plt.plot(data, ub, '--', color=color, alpha=0.4)
    plt.fill_between(data, lb, ub, facecolor=color, alpha=0.2)


# ECDF plotting
def get_ecdf_bands_undirected(data, alpha):
    n = len(data)
    if n == 0:
        return

    epsilon = math.sqrt((math.log(2 / alpha)) / (2 * n))

    edf = np.zeros(np.shape(data))
    for i in range(0, n):
        edf[i] = sum(np.where(data <= data[i], 1, 0)) / n

    lower_band = edf - epsilon
    lower_band[np.where(lower_band < 0)[0]] = 0

    upper_band = edf + epsilon
    upper_band[np.where(upper_band > 1)[0]] = 1

    return lower_band, upper_band


def add_ecdf_with_band_plot_undirected(data, label, color):
    data = np.array(data)
    data = data[~np.isnan(data)]

    data = np.sort(data)

    plt.step(data, np.arange(1, len(data) + 1) / np.float(len(data)), alpha=0.9, color=color,
             label='{0}: {1:.4f}'.format(label, np.mean(data)), lw=2)

    lb, ub = get_ecdf_bands_undirected(data, 0.05)
    plt.plot(data, lb, '--', color=color, alpha=0.4)
    plt.plot(data, ub, '--', color=color, alpha=0.4)
    plt.fill_between(data, lb, ub, facecolor=color, alpha=0.2)


def get_mean_ci(res, z_value, has_nan=False):
    if has_nan:
        return z_value * np.nanstd(res) / np.sqrt(np.sum(~np.isnan(res)))
    else:
        return z_value * np.std(res) / np.sqrt(len(res))
