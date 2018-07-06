import os
import math
import time
import pickle
import random
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

dataset_file_path = '/shared/DataSets/GooglePlus_Gong2012/raw/imc12/direct_social_structure.txt'
egonet_files_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/egonets-w-snapshots/'
nx2_incompatible_egonets_file_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/'
list_of_first_hop_nodes_file_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/edge-node-lists/' \
                                    'gplus-nodes-snap-0-list.pckl'

local_degree_empirical_results_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files-1/'

triad_ratio_empirical_results_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/' \
                                     'triad-link-formed-ratio/pickle-files/'
triad_ratio_empirical_plots_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/' \
                                   'triad-link-formed-ratio/plots/'

lp_results_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files-1/'


def read_entire_gplus_network():
    gplus_net = nx.DiGraph()

    with open(dataset_file_path) as infile:
        for l in infile:
            nums = l.split(" ")
            gplus_net.add_edge(int(nums[0]), int(nums[1]), snapshot=int(nums[2][0]))

    input("G+ network in...")

    return gplus_net


def get_gplus_ego_from_graph():
    gnet = nx.DiGraph()

    with open(dataset_file_path) as infile:
        for l in infile:
            nums = l.split(" ")
            gnet.add_edge(int(nums[0]), int(nums[1]), snapshot=int(nums[2][0]))

    input("G+ network in...")

    print("here")

    with open(list_of_first_hop_nodes_file_path, 'rb') as f:
        snap_0_nodes = pickle.load(f)

    ego_nodes = snap_0_nodes[:len(snap_0_nodes) / 2]
    # ego_nodes = snap_0_nodes[len(snap_0_nodes) / 2:]

    print("start working..")
    # Parallel(n_jobs=6)(delayed(gnet, gplus_get_egonet)(ego_node) for ego_node in ego_nodes)

    ctr = 0
    for ego_node in ego_nodes:

        # check if the egonet file already exists
        if os.path.isfile('{}{}.pckle'.format(nx2_incompatible_egonets_file_path, ego_node)):
            return

        ego_net = nx.ego_graph(gnet, ego_node, radius=2, center=True, undirected=True)

        print(type(ego_net))
        print("num nodes: {0}".format(nx.number_of_nodes(ego_net)))
        print("num edges: {0}".format(nx.number_of_edges(ego_net)))
        print("Preds: ")
        print(ego_net.predecessors(ego_node))
        print("Successors: ")
        print(ego_net.successors(ego_node))

        with open('{}{}.pckle'.format(nx2_incompatible_egonets_file_path, ego_node), 'wb') as f:
            pickle.dump([ego_node, ego_net], f, protocol=-1)

        ctr += 1
        print(ctr, end='\r')
        return


def gplus_get_egonet(gnet, ego_node):
    print("Fetching an egonet...")

    # check if the egonet file already exists
    if os.path.isfile('{}{}.pckle'.format(nx2_incompatible_egonets_file_path, ego_node)):
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

    with open('{}{}.pckle'.format(nx2_incompatible_egonets_file_path, ego_node), 'wb') as f:
        pickle.dump([ego_node, ego_net], f, protocol=-1)

    return


def gplus_get_all_nodes_appeared_in_snapshot(snapshot):
    nodes = set()

    cnt = 0
    with open(dataset_file_path) as infile:
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

    with open(list_of_first_hop_nodes_file_path, 'rb') as f:
        all_nodes = pickle.load(f)

    # ego_nodes = random.sample(all_nodes, n)
    ego_nodes = all_nodes[:math.ceil(len(all_nodes) / 2)]
    all_nodes = None

    # print("Selected {0} random nodes...".format(n))

    Parallel(n_jobs=6)(delayed(read_ego_gplus_graph)(ego_node) for ego_node in ego_nodes)


def read_ego_gplus_graph(ego_node):
    # check if the egonet file already exists
    if os.path.isfile('{}{}.pckle'.format(nx2_incompatible_egonets_file_path, ego_node)):
        return

    start_time = time.time()
    ego_net = nx.DiGraph()

    with open(dataset_file_path) as infile:
        for l in infile:
            nums = l.split(" ")
            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if ego_node == nums[0] or ego_node == nums[1]:
                ego_net.add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

    neighbors = ego_net.nodes()

    with open(dataset_file_path) as infile:
        for l in infile:
            nums = l.split(" ")
            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if nums[0] in neighbors or nums[1] in neighbors:
                ego_net.add_edge(nums[0], nums[1], snapshot=int(nums[2][0]))

    with open('{}{}.pckle'.format(nx2_incompatible_egonets_file_path, ego_node), 'wb') as f:
        pickle.dump([ego_node, ego_net], f, protocol=-1)

    print("network in! with {0} nodes and {1} edges.".format(len(ego_net.nodes()), len(ego_net.edges())))
    print("time -> {0} minutes".format((time.time() - start_time) / 60))


def read_ego_gplus_graph_by_batch_parallelizer(batch_size, n_process):
    print("Reading in Google+ data...")

    with open(list_of_first_hop_nodes_file_path, 'rb') as f:
        all_nodes = pickle.load(f)

    np.random.shuffle(all_nodes)

    Parallel(n_jobs=n_process)(delayed(read_ego_gplus_graph_by_batch)(ego_batch) for ego_batch in
                               batch(all_nodes, batch_size))


def read_ego_gplus_graph_by_batch(ego_nodes):
    ego_dict = {}
    for ego_node in ego_nodes:
        # check if the egonet file already exists
        if not os.path.isfile('{}{}.pckle'.format(egonet_files_path, ego_node)):
            ego_dict[ego_node] = nx.DiGraph()

    if len(ego_dict) == 0:
        return

    print("Start reading in the file for {0} nodes...".format(len(ego_dict)))
    start_time = time.time()

    with open(dataset_file_path) as infile:
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

    with open(dataset_file_path) as infile:
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

        with open('{}{}.pckle'.format(egonet_files_path, ego_node), 'wb') as f:
            pickle.dump([ego_node, ego_net_snapshots], f, protocol=-1)

    print("time -> {0} minutes".format((time.time() - start_time) / 60))


def create_gplus_multiple_egonets(n, batch_size):
    print("Reading in Google+ data...")

    with open(list_of_first_hop_nodes_file_path, 'rb') as f:
        all_nodes = pickle.load(f)

    ego_nodes = random.sample(all_nodes, n)

    for node in ego_nodes:
        if os.path.isfile('{}{}.pckle'.format(nx2_incompatible_egonets_file_path, node)):
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

    with open(dataset_file_path) as infile:
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

    with open(dataset_file_path) as infile:
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
        with open('{}{}.pckle'.format(nx2_incompatible_egonets_file_path, ego_node), 'wb') as f:
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


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]