import re
import os
import sys
import math
import time
import pickle
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

raw_gplus_path = "/shared/DataSets/GooglePlus_Gong2012/raw/imc12/direct_social_structure.txt"


def read_entire_gplus_network():
    gplus_net = nx.DiGraph()

    with open(raw_gplus_path) as infile:
        for l in infile:
            nums = l.split(" ")
            gplus_net.add_edge(int(nums[0]), int(nums[1]), snapshot=int(nums[2][0]))

    input("G+ network in...")

    return gplus_net


def get_gplus_ego_from_graph():
    gnet = nx.DiGraph()

    with open(raw_gplus_path) as infile:
        for l in infile:
            nums = l.split(" ")
            gnet.add_edge(int(nums[0]), int(nums[1]), snapshot=int(nums[2][0]))

    input("G+ network in...")

    print("here")

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/edge-node-lists/gplus-nodes-snap-0-list.pckl', 'rb') as f:
        snap_0_nodes = pickle.load(f)

    ego_nodes = snap_0_nodes[:len(snap_0_nodes) / 2]
    # ego_nodes = snap_0_nodes[len(snap_0_nodes) / 2:]

    print("start working..")
    # Parallel(n_jobs=6)(delayed(gnet, gplus_get_egonet)(ego_node) for ego_node in ego_nodes)

    ctr = 0
    for ego_node in ego_nodes:

        # check if the egonet file already exists
        if os.path.isfile(
                '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/{0}.pckle'.format(
                        ego_node)):
            return

        # check if the egonet file already exists
        if os.path.isfile(
                '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/new/{0}.pckle'.format(
                    ego_node)):
            return

        ego_net = nx.ego_graph(gnet, ego_node, radius=2, center=True, undirected=True)

        print(type(ego_net))
        print("num nodes: {0}".format(nx.number_of_nodes(ego_net)))
        print("num edges: {0}".format(nx.number_of_edges(ego_net)))
        print("Preds: ")
        print(ego_net.predecessors(ego_node))
        print("Successors: ")
        print(ego_net.successors(ego_node))

        with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/new/{0}.pckle'.format(
                ego_node), 'wb') as f:
            pickle.dump([ego_node, ego_net], f, protocol=-1)

        ctr += 1
        print(ctr, end='\r')
        return


def gplus_get_egonet(gnet, ego_node):
    print("Fetching an egonet...")

    # check if the egonet file already exists
    if os.path.isfile(
            '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/{0}.pckle'.format(
                ego_node)):
        return

    # check if the egonet file already exists
    if os.path.isfile(
            '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/new/{0}.pckle'.format(
                ego_node)):
        return

    print("start ")
    ego_net = nx.ego_graph(gnet, ego_node, radius=2, center=True, undirected=True)

    print(type(ego_net))
    print("num nodes: {0}".format(nx.number_of_nodes(ego_net)))
    print("num edges: {0}".format(nx.number_of_edges(ego_net)))
    print("Preds: ")
    print(ego_net.predecessors(ego_node))
    print("Successors: ")
    print(ego_net.successors(ego_node))

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/new/{0}.pckle'.format(
            ego_node), 'wb') as f:
        pickle.dump([ego_node, ego_net], f, protocol=-1)

    return


def gplus_get_all_nodes_appeared_in_snapshot(snapshot):
    nodes = set()

    cnt = 0
    with open(raw_gplus_path) as infile:
        for l in infile:

            # if cnt % 100000 == 0:
            #     print(cnt, end='\r')
            # cnt += 1

            nums = l.split(" ")
            nums[2] = int(nums[2])
            print(nums[2])
            if nums[2] == snapshot:
                nodes.update([int(nums[0]), int(nums[1])])
                print("added")

    node_list = list(nodes)
    nodes = None

    print('\nNumber of nodes in snapshot {0}: {1}'.format(snapshot, len(node_list)))

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/edge-node-lists/gplus-nodes-snap-{0}-list.pckl'.format(
            snapshot), 'wb') as f:
        pickle.dump(node_list, f, protocol=-1)

    return node_list


def read_gplus_ego_graph():
    print("Reading in Google+ data...")

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/edge-node-lists/gplus-nodes-snap-0-list.pckl', 'rb') as f:
        all_nodes = pickle.load(f)

    # ego_nodes = random.sample(all_nodes, n)
    ego_nodes = all_nodes[:math.ceil(len(all_nodes) / 2)]
    all_nodes = None

    # print("Selected {0} random nodes...".format(n))

    Parallel(n_jobs=6)(delayed(read_ego_gplus_graph)(ego_node) for ego_node in ego_nodes)


def read_ego_gplus_graph(ego_node):
    # check if the egonet file already exists
    if os.path.isfile('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/{0}.pckle'.format(
            ego_node)):
        return
    start_time = time.time()
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

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/{0}.pckle'.format(ego_node),
              'wb') as f:
        pickle.dump([ego_node, ego_net], f, protocol=-1)

    print("network in! with {0} nodes and {1} edges.".format(len(ego_net.nodes()), len(ego_net.edges())))
    print("time -> {0} minutes".format((time.time() - start_time) / 60))


def read_ego_gplus_graph_by_batch_parallelizer(batch_size, n_process):
    print("Reading in Google+ data...")

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/edge-node-lists/gplus-nodes-snap-0-list.pckl',
              'rb') as f:
        all_nodes = pickle.load(f)

    np.random.shuffle(all_nodes)

    Parallel(n_jobs=n_process)(delayed(read_ego_gplus_graph_by_batch)(ego_batch) for ego_batch in batch(all_nodes, batch_size))


def read_ego_gplus_graph_by_batch(ego_nodes):
    ego_dict = {}
    for ego_node in ego_nodes:
        # check if the egonet file already exists
        if not os.path.isfile('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/egonets-w-snapshots/'
                              '{0}.pckle'.format(ego_node)):
            ego_dict[ego_node] = nx.DiGraph()

    if len(ego_dict) == 0:
        return

    print("Start reading in the file for {0} nodes...".format(len(ego_dict)))
    start_time = time.time()

    with open(raw_gplus_path) as infile:
        for l in infile:
            nums = l.split(" ")
            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if nums[0] in ego_dict:
                ego_dict[nums[0]].add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

            if nums[1] in ego_dict:
                ego_dict[nums[1]].add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

    neigh_dict = {}
    all_neighbors = set()

    for ego_node in ego_dict:
        neigh_temp = set(ego_dict[ego_node].nodes())
        neigh_temp.remove(ego_node)
        neigh_dict[ego_node] = neigh_temp
        all_neighbors = all_neighbors.union(neigh_temp)

    with open(raw_gplus_path) as infile:
        for l in infile:
            nums = l.split(" ")
            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if nums[0] in all_neighbors or nums[1] in all_neighbors:
                for ego_node in ego_dict:
                    if nums[0] in neigh_dict[ego_node] or nums[1] in neigh_dict[ego_node]:
                        ego_dict[ego_node].add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

    for ego_node in ego_dict:
        ego_net_snapshots = []

        for r in range(0, 4):
            temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_dict[ego_node].edges(data=True) if d['snapshot'] <= r])
            ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

        with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/egonets-w-snapshots/{0}.pckle'.format(
                ego_node), 'wb') as f:
            pickle.dump([ego_node, ego_net_snapshots], f, protocol=-1)

    print("time -> {0} minutes".format((time.time() - start_time) / 60))


def create_gplus_multiple_egonets(n, batch_size):
    print("Reading in Google+ data...")

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/edge-node-lists/gplus-nodes-snap-0-list.pckl', 'rb') as f:
        all_nodes = pickle.load(f)

    ego_nodes = random.sample(all_nodes, n)

    for node in ego_nodes:
        if os.path.isfile('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/{0}.pckle'.format(node)):
            ego_nodes.remove(node)

    all_nodes = None

    print("Selected {0} random nodes...".format(len(ego_nodes)))

    Parallel(n_jobs=10)(delayed(read_multiple_ego_gplus_graphs)(ego_nodes[i * batch_size: i * batch_size + batch_size])
                        for i in range(0, int(len(ego_nodes) / batch_size)))


def read_multiple_ego_gplus_graphs(ego_node_list):
    start_time = time.time()

    egonets = {}
    ego_neighbors = {}
    all_neighbors = set()

    for ego_node in ego_node_list:
        egonets[ego_node] = nx.DiGraph()

    with open(raw_gplus_path) as infile:
        for l in infile:
            nums = l.split(" ")
            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if nums[0] in egonets:
                egonets[nums[0]].add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

            if nums[1] in egonets:
                egonets[nums[1]].add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

    for ego_node in egonets:
        temp = egonets[ego_node].nodes()
        temp.remove(ego_node)
        ego_neighbors[ego_node] = set(temp)
        all_neighbors = all_neighbors.union(temp)

    with open(raw_gplus_path) as infile:
        for l in infile:
            nums = l.split(" ")
            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if nums[0] in all_neighbors:
                for ego_node in egonets:
                    if nums[0] in ego_neighbors[ego_node]:
                        egonets[ego_node].add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

            if nums[1] in all_neighbors:
                for ego_node in egonets:
                    if nums[1] in ego_neighbors[ego_node]:
                        egonets[ego_node].add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

    for ego_node in egonets:
        with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/{0}.pckle'.format(
                ego_node), 'wb') as f:
            pickle.dump([ego_node, egonets[ego_node]], f, protocol=-1)

        print("ego node {0} has {1} nodes and {2} edges.".format(ego_node, nx.number_of_nodes(egonets[ego_node]),
                                                                 nx.number_of_edges(egonets[ego_node])))

    print("Generating {0} ego-nets took {1} minutes.".format(len(ego_node_list), (time.time() - start_time) / 60))


def read_ego_gplus_pickle(ego_node):
    pickle_files = ['gplus-edges-snap-0-list.pckl', 'gplus-edges-snap-1-list.pckl', 'gplus-edges-snap-2-list.pckl',
                    'gplus-edges-snap-3-list.pckl']

    ego_net = nx.DiGraph()

    for i in range(4):
        with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/edge-node-lists/%s' % pickle_files[i], 'rb') as f:
            snapshot_edges = pickle.load(f)

        for u, v, attr in snapshot_edges:
            if u == ego_node or v == ego_node:
                ego_net.add_edge(u, v, attr)

        neighbors = ego_net.nodes()

        for u, v, attr in snapshot_edges:
            if u in neighbors or v in neighbors:
                ego_net.add_edge(u, v, attr)

    with open('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/{0}.pckle'.format(ego_node), 'wb') as f:
        pickle.dump([ego_node, ego_net], f, protocol=-1)


def plot_formed_vs_not(formed, not_formed, xlabel, subtitle, overall_mean_formed, overall_mean_not_formed,
                       save_plot=False, save_path=''):
    fig = None
    if save_plot:
        fig = plt.figure(figsize=(15, 8), dpi=100)
    else:
        fig = plt.figure()

    n_row = math.ceil(len(formed) / 5)
    n_col = math.ceil(len(not_formed) / n_row)

    overall_means_formed = []
    overall_means_not_formed = []

    for i in range(len(formed)):
        formed_mean = np.mean(formed[i])
        not_formed_mean = np.mean(not_formed[i])

        overall_means_formed.append(formed_mean)
        overall_means_not_formed.append(not_formed_mean)
        p = fig.add_subplot(n_row, n_col, i + 1)

        p.hist(formed[i], color='r', alpha=0.8, weights=np.zeros_like(formed[i]) + 1. / len(formed[i]),
               label="FEM: {0:.2f}".format(formed_mean))

        p.hist(not_formed[i], color='b', alpha=0.5, weights=np.zeros_like(not_formed[i]) + 1. / len(not_formed[i]),
               label="NFEM: {0:.2f}".format(not_formed_mean))

        p.legend(loc='upper right')
        plt.ylabel('Relative Frequency')

        plt.xlabel(xlabel)
        plt.suptitle(subtitle)

    overall_mean_formed.append(np.mean(overall_means_formed))
    overall_mean_not_formed.append(np.mean(overall_means_not_formed))

    if not save_plot:
        plt.show()
    else:
        current_fig = plt.gcf()
        current_fig.savefig(save_path)

    plt.close(fig)


def get_t01_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t02_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)) - set(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t03_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)) - set(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t04_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)) - set(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)) - set(ego_net.predecessors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t05_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t06_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)) - set(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t07_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.predecessors(ego_node)) - set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t08_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.predecessors(ego_node)) - set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t09_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.predecessors(ego_node)) - set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)) - set(ego_net.predecessors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_combined_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)).union(ego_net.predecessors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


########## Links formed in triad ratio analysis ##############
def get_t01_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego.intersection(predecessors_of_the_ego)

    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t02_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego.intersection(predecessors_of_the_ego)
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)) - set(ego_net.predecessors(z))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t03_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego - predecessors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t04_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego - predecessors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)) - set(ego_net.predecessors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t05_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego.intersection(predecessors_of_the_ego)
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t06_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego - predecessors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t07_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = predecessors_of_the_ego - successors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t08_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = predecessors_of_the_ego - successors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t09_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = predecessors_of_the_ego - successors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)) - set(ego_net.predecessors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # remove all nodes in the second hop that have any edge with the ego
    second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def empirical_triad_links_formed_ratio(ego_net_file, data_file_base_path, result_file_base_path):
    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    # return if the egonet is on the currently being analyzed list
    if os.path.isfile(result_file_base_path + 'temp-analyses-start/' + ego_net_file):
        return

    with open(result_file_base_path + 'temp-analyses-start/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    triangle_type_func = {
        'T01': get_t01_type_second_hop_nodes,
        'T02': get_t02_type_second_hop_nodes,
        'T03': get_t03_type_second_hop_nodes,
        'T04': get_t04_type_second_hop_nodes,
        'T05': get_t05_type_second_hop_nodes,
        'T06': get_t06_type_second_hop_nodes,
        'T07': get_t07_type_second_hop_nodes,
        'T08': get_t08_type_second_hop_nodes,
        'T09': get_t09_type_second_hop_nodes,
    }

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
    if ego_net_snapshots[-1].number_of_nodes() > 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)

        return

    results = {
        'T01': {}, 'T02': {}, 'T03': {}, 'T04': {}, 'T05': {}, 'T06': {}, 'T07': {}, 'T08': {}, 'T09': {}
    }

    total_num_edges_formed = 0

    for t_type in results.keys():
        results[t_type]['num_edges_formed'] = []
        results[t_type]['num_nodes'] = []
        results[t_type]['num_second_hop_nodes'] = []

    results['total_num_nodes'] = []

    for i in range(len(ego_net_snapshots) - 1):
        results['total_num_nodes'].append(ego_net_snapshots[i].number_of_nodes())

        for triangle_type in triangle_type_func.keys():
            second_hop_nodes, num_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i], ego_node)

            # number of nodes in the second hop that ego started to follow in the next snapshot
            next_snapshot_successors_of_the_ego = set(ego_net_snapshots[i + 1].successors(ego_node))
            num_edges_formed = len(next_snapshot_successors_of_the_ego.intersection(second_hop_nodes))

            results[triangle_type]['num_edges_formed'].append(num_edges_formed)
            results[triangle_type]['num_nodes'].append(num_nodes)
            results[triangle_type]['num_second_hop_nodes'].append(len(second_hop_nodes))

            total_num_edges_formed += num_edges_formed

    if total_num_edges_formed > 0:
        with open(result_file_base_path + 'results/' + ego_net_file, 'wb') as f:
            pickle.dump(results, f, protocol=-1)

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    # remove temp analyze file
    os.remove(result_file_base_path + 'temp-analyses-start/' + ego_net_file)

    print("Analyzed ego net {0}".format(ego_net_file))


def empirical_triad_list_formed_ratio_results_plot(result_file_base_path, plot_save_path, gather_individual_results=False):
    triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']

    if gather_individual_results:
        fraction_of_all_formed_edges = {}
        edge_probability = {}

        for t in triangle_types:
            fraction_of_all_formed_edges[t] = []
            edge_probability[t] = []

        for result_file in os.listdir(result_file_base_path + 'results'):
            with open(result_file_base_path + 'results/' + result_file, 'rb') as f:
                egonet_results = pickle.load(f)

            temp_fraction = []

            for t_type in triangle_types:
                t_type_total_formed_edges = np.sum(egonet_results[t_type]['num_edges_formed'])
                temp_fraction.append(t_type_total_formed_edges)

                t_type_total_num_second_hop_nodes = sum(egonet_results[t_type]['num_second_hop_nodes'])
                if t_type_total_num_second_hop_nodes != 0:
                    edge_probability[t_type].append(t_type_total_formed_edges / t_type_total_num_second_hop_nodes)
                else:
                    edge_probability[t_type].append(0)

            total_links_formed = np.sum(temp_fraction)
            temp_fraction = np.array(temp_fraction) / total_links_formed

            for i, t_type in enumerate(triangle_types):
                fraction_of_all_formed_edges[t_type].append(temp_fraction[i])

        # Create directory if not exists
        if not os.path.exists(result_file_base_path + "cumulated_results"):
            os.makedirs(result_file_base_path + "cumulated_results")

        # Write data into a single file for fraction of all edges
        with open(result_file_base_path + "cumulated_results/fraction_of_all_formed_edges.pckle", 'wb') as f:
            pickle.dump(fraction_of_all_formed_edges, f, protocol=-1)

        # Write data into a single file for edge probability
        with open(result_file_base_path + "cumulated_results/edge_probability.pckle", 'wb') as f:
            pickle.dump(fraction_of_all_formed_edges, f, protocol=-1)
    else:
        with open(result_file_base_path + "cumulated_results/fraction_of_all_formed_edges.pckle", 'rb') as f:
            fraction_of_all_formed_edges = pickle.load(f)

        with open(result_file_base_path + "cumulated_results/edge_probability.pckle", 'rb') as f:
            edge_probability = pickle.load(f)

    plot_fraction_results = []
    plot_edge_prob_results = []
    for t_type in triangle_types:
        plot_fraction_results.append(np.mean(fraction_of_all_formed_edges[t_type]))
        plot_edge_prob_results.append(np.mean(edge_probability[t_type]))

    # plotting the fraction of edges
    plt.figure()
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.plot(np.arange(1, len(triangle_types) + 1), plot_fraction_results, color='b', marker='o')

    plt.ylabel('Fraction of All Edges', fontsize=22)
    plt.xlabel('Triad Type', fontsize=22)
    plt.tight_layout()
    current_fig = plt.gcf()
    plt.xticks(np.arange(1, len(triangle_types) + 1), triangle_types)
    current_fig.savefig('{0}/triad_fraction_of_formed_edges.pdf'.format(plot_save_path), format='pdf')
    plt.clf()

    # plotting the edge probability
    plt.figure()
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.plot(np.arange(1, len(triangle_types) + 1), plot_edge_prob_results, color='b', marker='o')

    plt.ylabel('Edge Probability', fontsize=22)
    plt.xlabel('Triad Type', fontsize=22)
    plt.tight_layout()
    current_fig = plt.gcf()
    plt.xticks(np.arange(1, len(triangle_types) + 1), triangle_types)
    current_fig.savefig('{0}/triad_edge_probability.pdf'.format(plot_save_path), format='pdf')
    plt.clf()

