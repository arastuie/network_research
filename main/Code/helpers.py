import re
import math
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Reading facebook data
def read_facebook_graph():
    file = open("../Data/facebook-links.txt", 'r')

    print("Reading the original graph...")

    original_graph = nx.Graph()

    for l in file:
        p = re.compile('\d+')
        nums = p.findall(l)

        nums[0] = int(nums[0])
        nums[1] = int(nums[1])

        if not original_graph.has_node(nums[0]):
            original_graph.add_node(nums[0])

        if not original_graph.has_node(nums[1]):
            original_graph.add_node(nums[1])

        if len(nums) == 2:
            nums.append(-1)
        else:
            nums[2] = int(nums[2])

        original_graph.add_edge(nums[0], nums[1], timestamp=nums[2])

    print("Original graph in.")
    return original_graph


def get_ego_centric_networks_in_fb(original_graph, n, hop=1, center=False):
    print("Generating the largest", n, "ego centric networks.")
    orig_snapshots = []

    oldest_timestamp = 1157454929
    seconds_in_90_days = 7776000
    for i in range(10):
        orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in original_graph.edges(data=True)
                                        if d['timestamp'] < (oldest_timestamp + seconds_in_90_days * i)]))

    orig_snapshots.append(original_graph)

    ego_centric_networks = []
    ego_nodes = []

    ego_centric_size = []
    min_index = 0
    for node in nx.nodes(orig_snapshots[0]):
        if len(ego_centric_networks) <= n:
            ego_nodes.append(node)
            ego_centric_size.append(len(orig_snapshots[0].neighbors(node)))
            min_index = -1
        elif len(orig_snapshots[0].neighbors(node)) > min(ego_centric_size):
            min_index = ego_centric_size.index(min(ego_centric_size))
            ego_nodes[min_index] = node
            ego_centric_size[min_index] = len(orig_snapshots[0].neighbors(node))
        else:
            continue

        ego_centric_network_snapshots = []
        for i in range(len(orig_snapshots)):
            ego_centric_network_snapshots.append(nx.ego_graph(orig_snapshots[i], node, radius=hop, center=center))

        if min_index == -1:
            ego_centric_networks.append(ego_centric_network_snapshots)
        else:
            ego_centric_networks[min_index] = ego_centric_network_snapshots

        # n -= 1
        # if n == 0:
        #     break

    with open('../Data/biggest_50_ego.pckl', 'wb') as f:
         pickle.dump([ego_centric_networks, ego_nodes], f, protocol=-1)

    return ego_centric_networks, ego_nodes


def plot_formed_vs_not(plot_type, formed, not_formed, plot_number, save_plot=False, save_path=''):
    fig = plt.figure()

    if save_plot:
        fig = plt.figure(figsize=(30, 15), dpi=200)

    n_row = math.ceil(len(formed) / 5)
    n_col = math.ceil(len(not_formed) / n_row)
    for i in range(len(formed)):
        formed_mean = np.mean(formed[i])
        not_formed_mean = np.mean(not_formed[i])

        p = fig.add_subplot(n_row, n_col, i + 1)

        p.hist(formed[i], color='r', alpha=0.8,
               weights=np.zeros_like(formed[i]) + 1. / len(formed[i]),
               label="FEM: {0:.2f}".format(formed_mean))

        p.hist(not_formed[i], color='b', alpha=0.5,
               weights=np.zeros_like(not_formed[i]) + 1. / len(not_formed[i]),
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

    if not save_plot:
        plt.show()
    else:
        current_fig = plt.gcf()
        current_fig.savefig(save_path)
    #fig.clf()
    #plt.clf()


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
        plt.suptitle('Total Number of Clusters All Common Neighbors Belong to (*) \n'
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
