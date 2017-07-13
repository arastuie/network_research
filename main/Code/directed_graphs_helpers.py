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

        p.hist(formed[i], color='r', alpha=0.8,
               weights=np.zeros_like(formed[i]) + 1. / len(formed[i]),
               label="FEM: {0:.2f}".format(formed_mean))

        p.hist(not_formed[i], color='b', alpha=0.5,
               weights=np.zeros_like(not_formed[i]) + 1. / len(not_formed[i]),
               label="NFEM: {0:.2f}".format(not_formed_mean))

        p.legend(loc='upper right')
        plt.ylabel('Relative Frequency')

        plt.xlabel(xlabel)
        plt.suptitle(subtitle)

    # overall_mean_formed.append(np.mean(overall_means_formed))
    # overall_mean_not_formed.append(np.mean(overall_means_not_formed))

    if not save_plot:
        plt.show()
    else:
        current_fig = plt.gcf()
        current_fig.savefig(save_path)

    plt.close(fig)


def get_t01_type_v_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if ego_net.has_edge(ego_node, v) or v == ego_node:
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t02_type_v_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)) - set(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if ego_net.has_edge(ego_node, v) or v == ego_node:
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t03_type_v_nodes(ego_net, ego_node):
    v_nodes = {}
    temp_z_nodes = set(ego_net.successors(ego_node)) - set(ego_net.predecessors(ego_node))
    for z in temp_z_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z))
        for v in temp_v_nodes:
            if ego_net.has_edge(ego_node, v):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    return v_nodes


def get_t05_type_v_nodes(ego_net, ego_node):
    v_nodes = {}
    temp_z_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    for z in temp_z_nodes:
        temp_v_nodes = set(ego_net.predecessors(z)) - set(ego_net.successors(z))
        for v in temp_v_nodes:
            if ego_net.has_edge(ego_node, v):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    return v_nodes


def get_t06_type_v_nodes(ego_net, ego_node):
    v_nodes = {}
    temp_z_nodes = set(ego_net.successors(ego_node)) - set(ego_net.predecessors(ego_node))
    for z in temp_z_nodes:
        temp_v_nodes = set(ego_net.predecessors(z)) - set(ego_net.successors(z))
        temp_v_nodes.remove(ego_node)
        for v in temp_v_nodes:
            if ego_net.has_edge(ego_node, v):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    return v_nodes
