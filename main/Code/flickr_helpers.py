import os
import time
import math
import pickle
import random
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
import directed_graphs_helpers as dh
import matplotlib.pyplot as plt

flickr_growth_file_path = '/shared/DataSets/FlickrGrowth/raw/flickr-growth.txt'
flickr_growth_empirical_result_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/flickr/pickle-files/'
flickr_growth_egonets_path = '/shared/DataSets/FlickrGrowth/egonets'


def read_graph():
    flickr_net = nx.DiGraph()
    t0 = time.time()
    with open(flickr_growth_file_path) as infile:
        for l in infile:
            nums = l.rstrip().split("\t")
            # there is one self-loop in the network that should be removed
            if nums[0] == nums[1]:
                continue

            t = int(time.mktime(time.strptime(nums[2], "%Y-%m-%d")))
            flickr_net.add_edge(int(nums[0]), int(nums[1]), timestamp=t)
    print("Flickr network in. Took {0:.2f}min".format((time.time() - t0) / 60))
    return flickr_net


def divide_to_snapshots(graph):
    # Here is the explanation for choosing these timestamps
    # Snapshot 0: All edges with timestamp 1162443600 (2006-11-02) since there are 17034806 of them
    # Snapshot 1: Between 1162530000 (2006-11-03) and 1165122000 (2006-12-03), since dates in Dec06 go up to 03 only
    # Snapshot 2: Between  1170478800 (2007-02-03) First date in data after (2006-12-03), and 1172638800 (2007-02-28)
    #       From this point on almost all other dates are available, so the snapshots have a 30 day length
    # Snapshot 3: Between 1172725200 (2007-03-01) and 1175313600 (2007-03-31)
    # Snapshot 4: Between 1175400000 (2007-04-01) and 1177905600 (2007-04-30)
    # Snapshot 5: Between 1177992000 (2007-05-01) and  1179460800 (2007-05-18), end of data
    snapshot_lengths = [(0, 1162443600), (1162530000, 1165122000), (1170478800, 1172638800), (1175400000, 1177905600),
                        (1177992000, 1179460800)]

    orig_snapshots = []

    # Goes up to len - 1 since the last snapshot is just the entire graph
    for i in range(len(snapshot_lengths) - 1):
        orig_snapshots.append(nx.DiGraph([(u, v, d) for u, v, d in graph.edges(data=True)
                                          if d['timestamp'] <= snapshot_lengths[i][1]]))

    orig_snapshots.append(graph)

    # print("Snapshots extracted!")
    return orig_snapshots


def read_snapshot_pickle_file():
    t0 = time.time()
    with open('/shared/DataSets/FlickrGrowth/flickr_growth_snapshots.pckl', 'rb') as f:
        snap_graph = pickle.load(f)

    print("Flickr pickle file in. Took {0:.2f}min".format((time.time() - t0) / 60))
    return snap_graph


def extract_ego_nets(n_egonets, max_n_nodes_in_egonet=100000):
    # Read in the pickle file of the snapshot flickr graph
    orig_snaps = read_snapshot_pickle_file()

    # Choosing which random nodes to extract from the first hop
    first_snap_nodes = list(orig_snaps[0].nodes())
    n_egonets_to_be_chosen = len(first_snap_nodes)
    if n_egonets > len(first_snap_nodes):
        n_egonets = len(first_snap_nodes)
        print("Not enough nodes to extract. Extracting {0} egonets...".format(n_egonets_to_be_chosen))
    elif n_egonets * 2 < len(first_snap_nodes):
        n_egonets_to_be_chosen = n_egonets * 2
    # select n_egonets_to_be_chosen to make sure you get n_egonets with the max number of nodes will not
    nodes_to_analyze = list(np.random.choice(first_snap_nodes, n_egonets_to_be_chosen))

    print()
    cnt = 0
    n_egonets_extracted = 0
    while n_egonets_extracted <= n_egonets and cnt < len(nodes_to_analyze):
        ego_net_snapshots = []
        for s in range(len(orig_snaps)):
            print("I'm here")
            ego_net_snapshots.append(nx.ego_graph(orig_snaps[s], nodes_to_analyze[cnt], radius=2, center=True,
                                                  undirected=True))
            print("but not here")
            if s == 0 and nx.number_of_nodes(ego_net_snapshots[0]) > max_n_nodes_in_egonet:
                cnt += 1
                break

        with open('{0}/{1}.pckl'.format(flickr_growth_egonets_path, nodes_to_analyze[cnt]), 'wb') as f:
            pickle.dump(ego_net_snapshots, f, protocol=-1)

        cnt += 1
        n_egonets_extracted += 1
        print("Progress: {0:2.2f}%".format(100 * n_egonets_extracted / n_egonets), end='\r')

    return


def create_gplus_multiple_egonets(n, batch_size, n_cores=10):
    print("Reading in Flickr data...")

    with open('/shared/DataSets/FlickrGrowth/first_snap_nodes_list.pckl', 'rb') as f:
        all_nodes = pickle.load(f)

    ego_nodes = random.sample(all_nodes, n)

    for node in ego_nodes:
        if os.path.isfile('{0}/{1}.pckl'.format(flickr_growth_egonets_path, node)):
            ego_nodes.remove(node)

    all_nodes = None

    print("Selected {0} random nodes...".format(len(ego_nodes)))

    Parallel(n_jobs=n_cores)(delayed(read_multiple_ego_flickr_graphs)
                             (ego_nodes[i * batch_size: i * batch_size + batch_size])
                             for i in range(0, int(len(ego_nodes) / batch_size)))


def read_multiple_ego_flickr_graphs(ego_node_list):
    start_time = time.time()

    egonets = {}
    ego_neighbors = {}
    all_neighbors = set()

    for ego_node in ego_node_list:
        egonets[ego_node] = nx.DiGraph()

    with open(flickr_growth_file_path) as infile:
        for l in infile:
            nums = l.rstrip().split("\t")

            # there is one self-loop in the network that should be removed
            if nums[0] == nums[1]:
                continue

            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if nums[0] in egonets or nums[1] in egonets:
                t = int(time.mktime(time.strptime(nums[2], "%Y-%m-%d")))

                if nums[0] in egonets:
                    egonets[nums[0]].add_edge(nums[0], nums[1], timestamp=t)

                if nums[1] in egonets:
                    egonets[nums[1]].add_edge(nums[0], nums[1], timestamp=t)

    for ego_node in egonets:
        temp = list(egonets[ego_node].nodes)
        temp.remove(ego_node)
        ego_neighbors[ego_node] = set(temp)
        all_neighbors = all_neighbors.union(temp)

    with open(flickr_growth_file_path) as infile:
        for l in infile:
            nums = l.rstrip().split("\t")

            # there is one self-loop in the network that should be removed
            if nums[0] == nums[1]:
                continue

            nums[0] = int(nums[0])
            nums[1] = int(nums[1])

            if nums[0] in all_neighbors or nums[1] in all_neighbors:
                t = int(time.mktime(time.strptime(nums[2], "%Y-%m-%d")))

                if nums[0] in all_neighbors:
                    for ego_node in egonets:
                        if nums[0] in ego_neighbors[ego_node]:
                            egonets[ego_node].add_edge(nums[0], nums[1], timestamp=t)

                if nums[1] in all_neighbors:
                    for ego_node in egonets:
                        if nums[1] in ego_neighbors[ego_node]:
                            egonets[ego_node].add_edge(nums[0], nums[1], timestamp=t)

    for ego_node in egonets:
        egonet_sanpshots = divide_to_snapshots(egonets[ego_node])
        with open('{0}/{1}.pckl'.format(flickr_growth_egonets_path, ego_node), 'wb') as f:
            pickle.dump([ego_node, egonet_sanpshots], f, protocol=-1)

    print("Generating {0} ego-nets took {1} minutes.".format(len(ego_node_list), (time.time() - start_time) / 60))
