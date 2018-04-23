import re
import time
import math
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

flickr_growth_file_path = '/shared/DataSets/FlickrGrowth/flickr-growth.txt'


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

    print("Snapshots extracted!")
    return orig_snapshots


def read_snapshot_pickle_file():
    t0 = time.time()
    with open('/shared/DataSets/FlickrGrowth/flickr_growth_snapshots.pckl', 'rb') as f:
        snap_graph = pickle.load(f)

    print("Flickr pickle file in. Took {0:.2f}".format((time.time() - t0) / 60))
    return snap_graph