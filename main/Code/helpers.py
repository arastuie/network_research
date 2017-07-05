import re
import sys
import math
import pickle
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


raw_gplus_path = "/shared/DataSets/GooglePlus_Gong2012/raw/imc12/direct_social_structure.txt"


def gplus_get_all_nodes_first_appeared(snapshot):
    nodes = set()

    cnt = 0
    with open(raw_gplus_path) as infile:
        for l in infile:

            if cnt % 100000 == 0:
                print(cnt, end='\r')
            cnt += 1

            nums = l.split(" ")
            nums[2] = int(nums[2])
            if nums[2] != snapshot:
                continue

            nodes.update([int(nums[0]), int(nums[1])])

    node_list = list(nodes)
    nodes = None

    print('\nNumber of nodes in snapshot {0}: {1}'.format(snapshot, len(node_list)))

    with open('../Data/gplus/gplus-nodes-snap-{0}-list.pckl'.format(snapshot), 'wb') as f:
        pickle.dump(node_list, f, protocol=-1)

    return node_list


def read_gplus_ego_graph(n):
    print("Reading in Google+ data...")

    with open('../Data/gplus/gplus-nodes-snap-0-list.pckl', 'rb') as f:
        all_nodes = pickle.load(f)

    ego_nodes = random.sample(all_nodes, n)
    all_nodes = None

    print("Selected {0} random nodes...".format(n))

    Parallel(n_jobs=15)(delayed(read_ego_gplus_graph)(ego_node) for ego_node in ego_nodes)


def read_ego_gplus_graph(ego_node):
    ego_net = nx.DiGraph()

    with open(raw_gplus_path) as infile:
        for l in infile:
            nums = l.split(" ")
            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if ego_node == nums[0] or ego_node == nums[1]:
                ego_net.add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

    neighbors = ego_net.nodes()

    with open(raw_gplus_path) as infile:
        for l in infile:
            nums = l.split(" ")
            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if nums[0] in neighbors or nums[1] in neighbors:
                ego_net.add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

    with open('../Data/gplus-ego/first-hop-nodes/{0}.pckle'.format(ego_node), 'wb') as f:
        pickle.dump([ego_node, ego_net], f, protocol=-1)

    print("network in! with {0} nodes and {1} edges.".format(len(ego_net.nodes()), len(ego_net.edges())))


def read_ego_gplus_pickle(ego_node):
    pickle_files = ['gplus-edges-snap-0-list.pckl', 'gplus-edges-snap-1-list.pckl', 'gplus-edges-snap-2-list.pckl',
                    'gplus-edges-snap-3-list.pckl']

    ego_net = nx.DiGraph()

    for i in range(4):
        with open('../Data/%s' % pickle_files[i], 'rb') as f:
            snapshot_edges = pickle.load(f)

        for u, v, attr in snapshot_edges:
            if u == ego_node or v == ego_node:
                ego_net.add_edge(u, v, attr)

        neighbors = ego_net.nodes()

        for u, v, attr in snapshot_edges:
            if u in neighbors or v in neighbors:
                ego_net.add_edge(u, v, attr)

    with open('../Data/gplus-ego/{0}.pckle'.format(ego_node), 'wb') as f:
        pickle.dump([ego_node, ego_net], f, protocol=-1)


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


def get_ego_centric_networks_in_fb(original_graph, n, pickle_file_name, search_type='random', hop=1, center=False):
    """
    Returns n ego centric networks out of divided into 10 snapshots
    
    :param original_graph: The original graph containing all nodes and edges
    :param n: Number of graphs ego centric networks wanted
    :param pickle_file_name: name of the pickle file to save the result. It will be saved at '../Data/pickle_file_name'.
    :param search_type: if 'random' will select n totally random nodes as the ego nodes
                 if 'absolute_biggest' will select the nodes with the biggest number of neighbors
                 if 'relative_biggest' will select the nodes with the biggest number of neighbors out of the first 
                    n * 4 nodes
    :param hop: Desired number of hops of the returned ego centric networks
    :param center: if True, the ego node will be included to the ego centric networks
    :return: 
    """
    if n > nx.number_of_nodes(original_graph):
        sys.exit("There are not enough nodes to generate %d ego centric networks." % n)

    if search_type != 'random' and search_type != 'absolute_biggest' and search_type != 'relative_biggest':
        sys.exit("search_type '%s' is not acceptable." % search_type)

    print("Generating %d %s ego centric networks." % (n, search_type))
    orig_snapshots = []

    oldest_timestamp = 1157454929
    seconds_in_90_days = 7776000
    for i in range(10):
        orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in original_graph.edges(data=True)
                                        if d['timestamp'] < (oldest_timestamp + seconds_in_90_days * i)]))

    orig_snapshots.append(original_graph)

    ego_centric_networks = []
    ego_nodes = []

    orig_nodes = nx.nodes(orig_snapshots[0])
    if search_type == 'random':
        np.random.shuffle(orig_nodes)
        ego_nodes = orig_nodes[0:n]

    elif search_type == 'relative_biggest' or search_type == 'absolute_biggest':
        max_node_search = n * 4
        if search_type == 'absolute_biggest' or max_node_search > len(orig_nodes):
            max_node_search = len(orig_nodes)

        ego_centric_size = []
        for i in range(max_node_search):
            if len(ego_centric_networks) < n:
                ego_nodes.append(orig_nodes[i])
                ego_centric_size.append(len(orig_snapshots[0].neighbors(orig_nodes[i])))
            elif len(orig_snapshots[0].neighbors(orig_nodes[i])) > min(ego_centric_size):
                min_index = ego_centric_size.index(min(ego_centric_size))
                ego_nodes[min_index] = orig_nodes[i]
                ego_centric_size[min_index] = len(orig_snapshots[0].neighbors(orig_nodes[i]))

    for node in ego_nodes:
        ego_centric_network_snapshots = []
        for i in range(len(orig_snapshots)):
            ego_centric_network_snapshots.append(nx.ego_graph(orig_snapshots[i], node, radius=hop, center=center))

        ego_centric_networks.append(ego_centric_network_snapshots)

    with open('../Data/%s' % pickle_file_name, 'wb') as f:
        pickle.dump([ego_centric_networks, ego_nodes], f, protocol=-1)

    return ego_centric_networks, ego_nodes


def plot_formed_vs_not(plot_type, formed, not_formed, plot_number, save_plot=False, save_path='', hop_number=1):
    fig = None

    if save_plot:
        fig = plt.figure(figsize=(30, 15), dpi=200)
    else:
        fig = plt.figure()

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

    avg = len(nx.neighbors(network, nodes[0]))
    count = 1
    for n in range(1, len(nodes)):
        avg = (avg * count + len(nx.neighbors(network, nodes[n]))) / (count + 1)
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
