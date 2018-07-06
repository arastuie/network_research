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


########## Local degree empirical analysis ##############
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


def run_local_degree_empirical_analysis(ego_net_file, results_base_path, egonet_file_base_path):
    # return if the egonet is on the analyzed list
    if os.path.isfile(results_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if os.path.isfile(results_base_path + 'skipped_egonets/' + ego_net_file):
        return

    # return if the egonet is on the currently being analyzed list
    if os.path.isfile(results_base_path + 'temp-analyses-start/' + ego_net_file):
        return

    triangle_type_func = {
        'T01': get_t01_type_nodes,
        'T02': get_t02_type_nodes,
        'T03': get_t03_type_nodes,
        'T04': get_t04_type_nodes,
        'T05': get_t05_type_nodes,
        'T06': get_t06_type_nodes,
        'T07': get_t07_type_nodes,
        'T08': get_t08_type_nodes,
        'T09': get_t09_type_nodes,
    }

    with open(egonet_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
    if nx.number_of_nodes(ego_net_snapshots[0]) > 100000:
        with open(results_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)

        return

    with open(results_base_path + 'temp-analyses-start/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

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

            # dict of lists -> z: [local_in, local_out, global_in, global_out]
            z_degree_info = {}

            for z in first_hop_nodes:
                z_preds = set(ego_net_snapshots[i].predecessors(z))
                z_succs = set(ego_net_snapshots[i].successors(z))

                z_degree_info[z] = [len(z_preds.intersection(first_hop_nodes)),
                                    len(z_succs.intersection(first_hop_nodes)),
                                    len(z_preds),
                                    len(z_succs)]

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
                    local_temp_in_degree.append(z_degree_info[z][0])
                    local_temp_out_degree.append(z_degree_info[z][1])
                    global_temp_in_degree.append(z_degree_info[z][2])
                    global_temp_out_degree.append(z_degree_info[z][3])

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

        with open(results_base_path + triangle_type + '/' + ego_net_file, 'wb') as f:
            pickle.dump([np.mean(local_snapshots_formed_z_in_degree),
                         np.mean(global_snapshots_formed_z_in_degree),
                         np.mean(local_snapshots_formed_z_out_degree),
                         np.mean(global_snapshots_formed_z_out_degree),
                         np.mean(local_snapshots_not_formed_z_in_degree),
                         np.mean(global_snapshots_not_formed_z_in_degree),
                         np.mean(local_snapshots_not_formed_z_out_degree),
                         np.mean(global_snapshots_not_formed_z_out_degree)], f, protocol=-1)

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(results_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    # remove temp analyze file
    os.remove(results_base_path + 'temp-analyses-start/' + ego_net_file)

    print("Analyzed ego net {0}".format(ego_net_file))


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

            # Calculating fraction ratio
            temp_snapshot_total_edges_formed = []

            for t_type in triangle_types:
                temp_snapshot_total_edges_formed.append(egonet_results[t_type]['num_edges_formed'])

            temp_snapshot_total_edges_formed = np.sum(temp_snapshot_total_edges_formed, axis=0)

            for t_type in triangle_types:
                temp_fraction = []
                for i in range(len(egonet_results[t_type]['num_edges_formed'])):
                    if temp_snapshot_total_edges_formed[i] != 0:
                        temp_fraction.append(egonet_results[t_type]['num_edges_formed'][i] /
                                             temp_snapshot_total_edges_formed[i])
                    else:
                        temp_fraction.append(0)

                fraction_of_all_formed_edges[t_type].append(np.mean(temp_fraction))


            # Calculating edge probability
            for t_type in triangle_types:
                temp_prob = []
                for i in range(len(egonet_results[t_type]['num_edges_formed'])):
                    if egonet_results[t_type]['num_second_hop_nodes'][i] != 0:
                        temp_prob.append(egonet_results[t_type]['num_edges_formed'][i] /
                                         egonet_results[t_type]['num_second_hop_nodes'][i])
                    else:
                        temp_prob.append(0)

                edge_probability[t_type].append(np.mean(temp_prob))

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
    plot_fraction_results_err = []
    plot_edge_prob_results = []
    plot_edge_prob_results_err = []
    for t_type in triangle_types:
        plot_fraction_results.append(np.mean(fraction_of_all_formed_edges[t_type]))
        plot_fraction_results_err.append(np.std(fraction_of_all_formed_edges[t_type]) /
                                         np.sqrt(len(fraction_of_all_formed_edges[t_type])))

        plot_edge_prob_results.append(np.mean(edge_probability[t_type]))
        plot_edge_prob_results_err.append(np.std(edge_probability[t_type]) /
                                          np.sqrt(len(edge_probability[t_type])))

    # plotting the fraction of edges
    plt.figure()
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.errorbar(np.arange(1, len(triangle_types) + 1), plot_fraction_results, yerr=plot_fraction_results_err,
                 color='b', fmt='--o')
    plt.ylabel('Fraction of All Edges', fontsize=22)
    plt.xlabel('Triad Type', fontsize=22)
    plt.tight_layout()
    current_fig = plt.gcf()
    plt.xticks(np.arange(1, len(triangle_types) + 1), triangle_types)
    current_fig.savefig('{0}triad_fraction_of_formed_edges.pdf'.format(plot_save_path), format='pdf')
    plt.clf()

    # plotting the edge probability
    plt.figure()
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.errorbar(np.arange(1, len(triangle_types) + 1), plot_edge_prob_results, yerr=plot_edge_prob_results_err,
                 color='b', fmt='--o')

    plt.ylabel('Edge Probability', fontsize=22)
    plt.xlabel('Triad Type', fontsize=22)
    plt.tight_layout()
    current_fig = plt.gcf()
    plt.xticks(np.arange(1, len(triangle_types) + 1), triangle_types)
    current_fig.savefig('{0}triad_edge_probability.pdf'.format(plot_save_path), format='pdf')
    plt.clf()

