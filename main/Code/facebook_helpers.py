import os
import time
import math
import pickle
import networkx as nx
import numpy as np


dataset_file_path = '/shared/DataSets/FacebookViswanath2009/raw/facebook-links.txt'
egonet_files_path = '/shared/DataSets/FacebookViswanath2009/egocentric/all_egonets/'
empirical_pickle_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/pickle-files-1/'


# ********** Reading facebook data ********** #
def read_graph():
    print("Reading the original graph...")

    t0 = time.time()
    original_graph = nx.Graph()

    with open(dataset_file_path, 'r') as infile:
        for l in infile:
            nums = l.rstrip().split("\t")

            # replace no timestamp with -1
            if nums[2] == '\\N':
                nums[2] = -1

            original_graph.add_edge(int(nums[0]), int(nums[1]), timestamp=int(nums[2]))

    print("Facebook network in. Took {0:.2f}min".format((time.time() - t0) / 60))
    return original_graph


def extract_all_ego_centric_networks_in_fb(original_graph):
    """
    Extracts and saves all ego centric networks, divided into 10 snapshots

    :param original_graph: The original graph containing all nodes and edges
    """
    print()

    orig_snapshots = []
    first_timestamp = 1157454929
    seconds_in_90_days = 90 * 24 * 60 * 60

    for i in range(1, 11):
        orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in original_graph.edges(data=True)
                                        if d['timestamp'] < (first_timestamp + seconds_in_90_days * i)]))

    extracted_nodes = set()

    # Nodes appeared only in the last snapshots do not need to be extracted
    for i in range(len(orig_snapshots) - 1):
        nodes_to_extract = set(orig_snapshots[i].nodes()) - extracted_nodes

        for ego in nodes_to_extract:
            ego_snapshots = []

            for u in range(i, len(orig_snapshots)):
                ego_snapshots.append(nx.ego_graph(orig_snapshots[u], ego, radius=2))

            with open("{}{}.pckl".format(egonet_files_path, ego), 'wb') as f:
                pickle.dump([ego, ego_snapshots], f, protocol=-1)

            extracted_nodes.add(ego)
            print("Num nodes extracted: {0}".format(len(extracted_nodes)), end="\r")

    return


# ********** Local degree empirical analysis ********** #
def run_local_degree_empirical_analysis(ego_net_file):
    # This analysis is separated into befor and after PYMK. FB intorduced PYMK around March 2008, which means
    # links created between snapshots 0 to 5 are for before PYMK and the ones made between 5 to 9 are after.
    # Also since not all ego nets have 10 snapshots, the index for before and after must be first calculated.

    # return if the egonet is on the analyzed list in after pymk, it must be in before as well
    if os.path.isfile(empirical_pickle_path + 'after-pymk/analyzed_egonets/' + ego_net_file):
        return

    with open(egonet_files_path + ego_net_file, 'rb') as f:
        ego, ego_snaps = pickle.load(f)

    num_snaps = len(ego_snaps)

    for pymk_type in ['before-pymk', 'after-pymk']:
        if pymk_type == 'before-pymk':
            # at least need 6 snapshots to have 1 snapshot jump for before PYMK
            before_pymk_ending_snap = num_snaps - 5
            if before_pymk_ending_snap < 1:
                continue

            start = 0
            end = before_pymk_ending_snap

        else:
            after_pymk_starting_snap = num_snaps - 5
            if after_pymk_starting_snap < 0:
                after_pymk_starting_snap = 0

            start = after_pymk_starting_snap
            end = num_snaps - 1

        snapshots_local_formed = []
        snapshots_global_formed = []
        snapshots_local_not_formed = []
        snapshots_global_not_formed = []

        for i in range(start, end):
            # Get a list of v nodes which did or did not form an edge with the ego in the next snapshot
            current_snap_z_nodes = set(ego_snaps[i].neighbors(ego))
            current_snap_v_nodes = set(ego_snaps[i].nodes()) - current_snap_z_nodes
            current_snap_v_nodes.remove(ego)

            next_snap_z_nodes = set(ego_snaps[i + 1].neighbors(ego))

            formed_v_nodes = next_snap_z_nodes.intersection(current_snap_v_nodes)
            not_formed_v_nodes = current_snap_v_nodes - formed_v_nodes

            if len(formed_v_nodes) == 0 or len(not_formed_v_nodes) == 0:
                continue

            len_first_hop = len(current_snap_z_nodes)
            tot_num_nodes = ego_snaps[i].number_of_nodes()

            z_local_formed = []
            z_global_formed = []
            z_local_not_formed = []
            z_global_not_formed = []

            # Getting degrees of the z nodes to speed up the process
            z_local_degree = {}
            z_global_degree = {}
            for z in current_snap_z_nodes:
                z_local_degree[z] = len(list(nx.common_neighbors(ego_snaps[i], z, ego)))
                z_global_degree[z] = nx.degree(ego_snaps[i], z)

            for v in formed_v_nodes:
                common_neighbors = nx.common_neighbors(ego_snaps[i], v, ego)
                temp_local = []
                temp_global = []

                for cn in common_neighbors:
                    temp_local.append(z_local_degree[cn])
                    temp_global.append(z_global_degree[cn])

                z_local_formed.append(np.mean(temp_local))
                z_global_formed.append(np.mean(temp_global))

            for v in not_formed_v_nodes:
                common_neighbors = list(nx.common_neighbors(ego_snaps[i], v, ego))
                temp_local = []
                temp_global = []

                for cn in common_neighbors:
                    temp_local.append(z_local_degree[cn])
                    temp_global.append(z_global_degree[cn])

                z_local_not_formed.append(np.mean(temp_local))
                z_global_not_formed.append(np.mean(temp_global))

            snapshots_local_formed.append(np.mean(z_local_formed) / len_first_hop)
            snapshots_global_formed.append(np.mean(z_global_formed) / tot_num_nodes)
            snapshots_local_not_formed.append(np.mean(z_local_not_formed) / len_first_hop)
            snapshots_global_not_formed.append(np.mean(z_global_not_formed) / tot_num_nodes)

        if len(snapshots_local_formed) > 0:
            with open(empirical_pickle_path + pymk_type + '/results/' + ego_net_file, 'wb') as f:
                pickle.dump([np.mean(snapshots_local_formed),
                             np.mean(snapshots_global_formed),
                             np.mean(snapshots_local_not_formed),
                             np.mean(snapshots_global_not_formed)], f, protocol=-1)

        # save an empty file in analyzed_egonets to know which ones were analyzed
        with open(empirical_pickle_path + pymk_type + '/analyzed-egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)

    print("Analyzed ego net {0}".format(ego_net_file))
    return


# ********** Link prediction analysis ********** #
def calc_top_k_scores(y_scores, y_true, top_k_values, percent_score):
    index_of_top_k_scores = np.argsort(y_scores)[::-1][:top_k_values[-1]]
    top_preds = y_true[index_of_top_k_scores]
    for k in top_k_values:
        percent_score[k].append(sum(top_preds[:k]) / k)


def common_neighbors_index(ego_net, non_edges):
    scores = []

    for u, v in non_edges:
        scores.append(len(list(nx.common_neighbors(ego_net, u, v))))

    return scores


def degree_corrected_common_neighbors_index(ego_net, non_edges, first_hop_nodes):
    scores = []

    for u, v in non_edges:
        first_hop_degrees = []
        common_neighbors = nx.common_neighbors(ego_net, u, v)
        # v_node_neighbors = set(nx.neighbors(ego_net, v))

        for c in common_neighbors:
            cn_neighbors = set(nx.neighbors(ego_net, c))

            # total degree
            # t = len(cn_neighbors)

            # local degree
            x = len(cn_neighbors.intersection(first_hop_nodes)) + 2

            # # total degree - local degree
            # y = t - x

            first_hop_degrees.append(math.log(x))

        first_hop_degree_index = sum(first_hop_degrees)

        scores.append(first_hop_degree_index)

    return scores


def degree_corrected_adamic_adar_index(ego_net, non_edges, first_hop_nodes):
    scores = []

    for u, v in non_edges:
        first_hop_degrees = []
        # other_degrees = []
        common_neighbors = nx.common_neighbors(ego_net, u, v)
        v_node_neighbors = set(nx.neighbors(ego_net, v))

        for c in common_neighbors:
            cn_neighbors = set(nx.neighbors(ego_net, c))
            # x = len(cn_neighbors.intersection(first_hop_nodes))

            # total degree
            t = len(cn_neighbors)

            # local degree
            x = len(cn_neighbors.intersection(first_hop_nodes))

            # total degree - local degree
            y = t - x

            if x == 0:
                x = 1

            # if y == 0:
            #     y = 0.5

            # score = (x ** 3 + y ** 3) / (x * y)
            # print(x, y, len(cn_neighbors))
            # if score <= 1:
            #     print(score)

            # first_hop_degrees.append(x ** 2 / (len(cn_neighbors) * len(first_hop_nodes)))

            first_hop_degrees.append((x * (1 - x / t)) + (y * (t / x)))
            # other_degrees.append(len(cn_neighbors))

        # for i in range(len(first_hop_degrees)):
        #     if first_hop_degrees[i] == 0:
        #         first_hop_degrees[i] = 1.33
        #     elif first_hop_degrees[i] == 1:
        #         first_hop_degrees[i] = 1.66

        # other_degrees_index = sum((math.log(d) * -1) for d in other_degrees)
        first_hop_degree_index = sum(1 / math.log(d) for d in first_hop_degrees)
        # first_hop_degree_index = sum(first_hop_degrees)
        scores.append(first_hop_degree_index)

    return scores


def cclp(ego_net, non_edges):
    scores = []

    for u, v in non_edges:
        common_neighbors = nx.common_neighbors(ego_net, u, v)
        score = 0

        for c in common_neighbors:
            c_tri = nx.triangles(ego_net, c)
            c_deg = ego_net.degree(c)

            score += c_tri / (c_deg * (c_deg - 1) / 2)

        scores.append(score)

    return scores


def car(ego_net, non_edges):
    scores = []

    for u, v in non_edges:
        common_neighbors = list(nx.common_neighbors(ego_net, u, v))
        cc_sub_g = ego_net.subgraph(common_neighbors)

        scores.append(len(common_neighbors) * cc_sub_g.number_of_edges())

    return scores


def run_link_prediction_analysis(ego_snapshots, ego_node, top_k_values, snap_range, scores_list):
    percent_scores = {}
    for score in scores_list:
        percent_scores[score] = {}
        for k in top_k_values:
            percent_scores[score][k] = []

    for i in snap_range:
        first_hop_nodes = set(ego_snapshots[i].neighbors(ego_node))

        if len(first_hop_nodes) == 0:
            continue

        second_hop_nodes = set(ego_snapshots[i].nodes()) - first_hop_nodes
        second_hop_nodes.remove(ego_node)

        formed_nodes = second_hop_nodes.intersection(ego_snapshots[i + 1].neighbors(ego_node))

        if len(formed_nodes) == 0:
            continue

        non_edges = []
        y_true = []

        for n in second_hop_nodes:
            # adding node with no edge as tuple
            non_edges.append((ego_node, n))

            if n in formed_nodes:
                y_true.append(1)
            else:
                y_true.append(0)

        # numpy array is needed for sorting purposes
        y_true = np.array(y_true)

        # evaluating different link prediction methods
        if 'cn' in percent_scores:
            y_scores = common_neighbors_index(ego_snapshots[i], non_edges)
            calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['cn'])

        if 'aa' in percent_scores:
            y_scores = [p for u, v, p in nx.adamic_adar_index(ego_snapshots[i], non_edges)]
            calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['aa'])

        if 'dccn' in percent_scores:
            y_scores = degree_corrected_common_neighbors_index(ego_snapshots[i], non_edges, first_hop_nodes)
            calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['dccn'])

        if 'dcaa' in percent_scores:
            y_scores = degree_corrected_adamic_adar_index(ego_snapshots[i], non_edges, first_hop_nodes)
            calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['dcaa'])

        if 'car' in percent_scores:
            y_scores = car(ego_snapshots[i], non_edges)
            calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['car'])

        if 'cclp' in percent_scores:
            y_scores = cclp(ego_snapshots[i], non_edges)
            calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['cclp'])

    return percent_scores
