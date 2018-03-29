import os
import math
import pickle
import numpy as np
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import directed_graphs_helpers as dh
from sklearn import metrics, preprocessing


def run_adamic_adar_on_ego_net(ego_snapshots, ego_node):
    lp_result = {
        'adamic_adar': {
            # 'fpr': [],
            # 'tpr': [],
            # 'precision': [],
            # 'recall': [],
            'auroc': [],
            'average_precision': []
        },
        'degree_corrected_adamic_adar': {
            # 'fpr': [],
            # 'tpr': [],
            # 'precision': [],
            # 'recall': [],
            'auroc': [],
            'average_precision': []
        },
        'common_neighbors': {
            'auroc': [],
            'average_precision': []
        },
        'degree_corrected_common_neighbors': {
            'auroc': [],
            'average_precision': []
        },
        # 'jaccard_coefficient': {
        #     'auroc': [],
        #     'average_precision': []
        # },
        # 'degree_corrected_jaccard_coefficient': {
        #     'auroc': [],
        #     'average_precision': []
        # }
    }
    
    for i in range(len(ego_snapshots) - 1):
        first_hop_nodes = set(ego_snapshots[i].neighbors(ego_node))

        if len(first_hop_nodes) < 30:
            continue

        second_hop_nodes = set(ego_snapshots[i].nodes()) - first_hop_nodes
        second_hop_nodes.remove(ego_node)

        formed_nodes = second_hop_nodes.intersection(ego_snapshots[i + 1].neighbors(ego_node))

        if len(formed_nodes) == 0 or len(second_hop_nodes) == 0:
            continue

        non_edges = []
        y_true = []

        for n in second_hop_nodes:
            # adding node tuple for adamic adar
            non_edges.append((ego_node, n))

            if n in formed_nodes:
                y_true.append(1)
            else:
                y_true.append(0)

        # Adamic Adar method
        y_scores_aa = [p for u, v, p in nx.adamic_adar_index(ego_snapshots[i], non_edges)]
        y_scores_aa = np.array(y_scores_aa).astype(float).reshape(1, -1)
        y_scores_aa = list(preprocessing.normalize(y_scores_aa, norm='max')[0])
        evaluate_prediction_result(y_true, y_scores_aa, lp_result['adamic_adar'])

        # DCAA method
        y_scores_dcaa = degree_corrected_adamic_adar_index(ego_snapshots[i], non_edges, first_hop_nodes)
        # y_scores_dcaa = shift_up(y_scores_dcaa)
        y_scores_dcaa = np.array(y_scores_dcaa).astype(float).reshape(1, -1)
        y_scores_dcaa = list(preprocessing.normalize(y_scores_dcaa, norm='max')[0])
        evaluate_prediction_result(y_true, y_scores_dcaa, lp_result['degree_corrected_adamic_adar'])

        # common neighbors method
        y_scores_cn = common_neighbors_index(ego_snapshots[i], non_edges)
        y_scores_cn = np.array(y_scores_cn).astype(float).reshape(1, -1)
        y_scores_cn = list(preprocessing.normalize(y_scores_cn, norm='max')[0])
        evaluate_prediction_result(y_true, y_scores_cn, lp_result['common_neighbors'])

        # DC common neighbors method
        y_scores_dccn = degree_corrected_common_neighbors_index(ego_snapshots[i], non_edges, first_hop_nodes)
        y_scores_dccn = np.array(y_scores_dccn).astype(float).reshape(1, -1)
        y_scores_dccn = list(preprocessing.normalize(y_scores_dccn, norm='max')[0])
        evaluate_prediction_result(y_true, y_scores_dccn, lp_result['degree_corrected_common_neighbors'])

        # # jaccard coefficient method
        # y_scores_jc = [p for u, v, p in nx.jaccard_coefficient(ego_snapshots[i], non_edges)]
        # y_scores_jc = np.array(y_scores_jc).astype(float).reshape(1, -1)
        # y_scores_jc = list(preprocessing.normalize(y_scores_jc, norm='max')[0])
        # evaluate_prediction_result(y_true, y_scores_jc, lp_result['jaccard_coefficient'])
        #
        # # DC jaccard coefficient method
        # y_scores_dcjc = degree_corrected_common_neighbors_index(ego_snapshots[i], non_edges, first_hop_nodes)
        # y_scores_dcjc = np.array(y_scores_dcjc).astype(float).reshape(1, -1)
        # y_scores_dcjc = list(preprocessing.normalize(y_scores_dcjc, norm='max')[0])
        # evaluate_prediction_result(y_true, y_scores_dcjc, lp_result['degree_corrected_jaccard_coefficient'])

    if len(lp_result['adamic_adar']['auroc']) == 0:
        return

    for method in lp_result:
        for score in lp_result[method]:
            lp_result[method][score] = np.mean(lp_result[method][score])

    return lp_result


def evaluate_prediction_result(y_true, y_scores, lp_result):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    # precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    average_precision = metrics.average_precision_score(y_true, y_scores)

    # lp_result['fpr'].append(fpr)
    # lp_result['tpr'].append(tpr)
    # lp_result['precision'].append(precision)
    # lp_result['recall'].append(recall)
    lp_result['auroc'].append(auroc)
    lp_result['average_precision'].append(average_precision)

    return


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


def test_lp_method(ego_net, non_edges, first_hop_nodes):
    scores = []
    for u, v in non_edges:
        common_neighbors = nx.common_neighbors(ego_net, u, v)
        temp_score = 0
        for c in common_neighbors:
            all_neighbors = set(nx.neighbors(ego_net, c))
            local_degree = len(all_neighbors.intersection(first_hop_nodes)) + 1
            global_degree = len(all_neighbors)

            temp_score += local_degree / math.log(global_degree)

        scores.append(temp_score)

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


def degree_corrected_jaccard_coefficient_index(ego_net, non_edges, first_hop_nodes):
    scores = []

    for u, v in non_edges:
        first_hop_degrees = []
        common_neighbors = nx.common_neighbors(ego_net, u, v)
        v_node_neighbors = set(nx.neighbors(ego_net, v))

        for c in common_neighbors:
            cn_neighbors = set(nx.neighbors(ego_net, c))

            # total degree
            # t = len(cn_neighbors)

            # local degree
            x = len(cn_neighbors.intersection(first_hop_nodes)) + 2

            # # total degree - local degree
            # y = t - x

            first_hop_degrees.append(math.log(x))

        first_hop_degree_index = sum(first_hop_degrees) / (len(v_node_neighbors.union(first_hop_nodes)))

        scores.append(first_hop_degree_index)

    return scores


def double_degree_adamic_adar_index(ego_net, non_edges, first_hop_nodes):
    scores = []
    for u, v in non_edges:
        common_neighbors = nx.common_neighbors(ego_net, u, v)
        double_degree_aa = 0
        for c in common_neighbors:
            cn_neighbors = set(nx.neighbors(ego_net, c))
            x = cn_neighbors.intersection(first_hop_nodes)
            y = cn_neighbors - first_hop_nodes
            if len(x) != 0:
                double_degree_aa += sum(1 / math.log(d) for d in ego_net.degree(x).values())

            if len(y) != 0:
                double_degree_aa -= sum(math.log(d) for d in ego_net.degree(y).values())

        scores.append(double_degree_aa)

    return scores


def common_neighbor_hop(ego_net, non_edges, first_hop_nodes, formed_nodes):
    degree_formation = []
    for u, v in non_edges:
        common_neighbors = nx.common_neighbors(ego_net, u, v)
        has_formed = int(v in formed_nodes)

        for c in common_neighbors:
            cn_neighbors = set(nx.neighbors(ego_net, c))
            x = len(cn_neighbors.intersection(first_hop_nodes))

            degree_formation.append([x / len(first_hop_nodes), len(cn_neighbors) / nx.number_of_nodes(ego_net),
                                     has_formed])

    return degree_formation


def plot_auroc_hist(lp_results):
    aurocs = {}
    for lp_method in lp_results[0]:
        aurocs[lp_method] = []

    for lp_result in lp_results:
        for lp_method in lp_result:
            aurocs[lp_method].append(lp_result[lp_method]['auroc'])

    for lp_method in aurocs:
        print("Mean AUROC of {0}: {1}".format(lp_method, np.mean(aurocs[lp_method])))

    print()

    plt.figure()
    bins = np.linspace(0, 1, 100)
    for lp_method in aurocs:
        plt.hist(aurocs[lp_method], bins, label='%s' % lp_method, alpha=0.5)
    plt.ylabel('Frequency')
    plt.xlabel('Area Under ROC')
    plt.title('AUROC Histogram of Ego Centric Graphs')
    plt.legend(loc="upper left")
    plt.show()


def plot_pr_hist(lp_results):
    aupr = {}
    for lp_method in lp_results[0]:
        aupr[lp_method] = []

    for lp_result in lp_results:
        for lp_method in lp_result:
            aupr[lp_method].append(lp_result[lp_method]['average_precision'])

    for lp_method in aupr:
        print("Mean AUPR of {0}: {1}".format(lp_method, np.mean(aupr[lp_method])))

    print()

    plt.figure()
    bins = np.linspace(0, 1, 100)
    for lp_method in aupr:
        plt.hist(aupr[lp_method], bins, label='%s' % lp_method, alpha=0.5)
    plt.ylabel('Frequency')
    plt.xlabel('Area Under Precision Recall (Average Precision)')
    plt.title('AUPR Histogram of Ego Centric Graphs')
    plt.legend(loc="upper right")
    plt.show()


def plot_degree_scatter(degree_formation):
    formed = degree_formation[np.where(degree_formation[:, 2] == 1)[0]]
    not_formed = degree_formation[np.where(degree_formation[:, 2] == 0)[0]]
    plt.scatter(formed[:, 0], formed[:, 1], label="Formed", color='green', marker='*', s=100)
    plt.scatter(not_formed[:, 0], not_formed[:, 1], label="Not Formed", color='red', marker='o', s=5, alpha=0.1)
    plt.xlabel("Normalized AA")
    plt.ylabel("Normalized Sum Ln(degree)")
    plt.show()
    plt.close()


def shift_up(arr):
    arr = np.array(arr)
    arr_min = arr.min()

    if arr_min >= 0:
        return arr

    arr += abs(arr_min)

    return arr


def run_adamic_adar_on_ego_net_ranking(ego_snapshots, ego_node, top_k_values, snap_range):

    percent_aa = {}
    percent_dcaa = {}
    percent_cn = {}
    percent_dccn = {}

    for k in top_k_values:
        percent_aa[k] = []
        percent_dcaa[k] = []
        percent_cn[k] = []
        percent_dccn[k] = []

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
            # adding node tuple for adamic adar
            non_edges.append((ego_node, n))

            if n in formed_nodes:
                y_true.append(1)
            else:
                y_true.append(0)

        y_scores_aa = [p for u, v, p in nx.adamic_adar_index(ego_snapshots[i], non_edges)]
        y_scores_dcaa = degree_corrected_adamic_adar_index(ego_snapshots[i], non_edges, first_hop_nodes)

        y_scores_cn = common_neighbors_index(ego_snapshots[i], non_edges)
        y_scores_dccn = degree_corrected_common_neighbors_index(ego_snapshots[i], non_edges, first_hop_nodes)

        combo_scores = np.concatenate((np.array(y_scores_aa).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dcaa).astype(float).reshape(-1, 1),
                                       np.array(y_scores_cn).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dccn).astype(float).reshape(-1, 1),
                                       np.array(y_true).reshape(-1, 1)), axis=1)

        combo_scores_aa_sorted = combo_scores[combo_scores[:, 0].argsort()[::-1]]
        combo_scores_dcaa_sorted = combo_scores[combo_scores[:, 1].argsort()[::-1]]

        combo_scores_cn_sorted = combo_scores[combo_scores[:, 2].argsort()[::-1]]
        combo_scores_dccn_sorted = combo_scores[combo_scores[:, 3].argsort()[::-1]]

        # ones_index_aa = np.where(combo_scores_aa_sorted[:, 2] == 1)[0]
        # ones_index_dcaa = np.where(combo_scores_dcaa_sorted[:, 2] == 1)[0]
        # ones_aa = combo_scores_aa_sorted[ones_index_aa]
        # ones_dcaa = combo_scores_dcaa_sorted[ones_index_dcaa]

        # top_n = math.ceil(len(y_true) * 0.03)
        for k in percent_aa.keys():
            percent_aa[k].append(sum(combo_scores_aa_sorted[:k, -1]) / k)
            percent_dcaa[k].append(sum(combo_scores_dcaa_sorted[:k, -1]) / k)
            percent_cn[k].append(sum(combo_scores_cn_sorted[:k, -1]) / k)
            percent_dccn[k].append(sum(combo_scores_dccn_sorted[:k, -1]) / k)

        # ones_index_aa = ones_index_aa / len(y_true)
        # ones_index_dcaa = ones_index_dcaa / len(y_true)
        #
        # for m in ones_index_aa:
        #     percent_aa.append(m)
        #
        # for m in ones_index_dcaa:
        #     percent_dcaa.append(m)

    return percent_aa, percent_dcaa, percent_cn, percent_dccn


def run_adamic_adar_on_ego_net_ranking_with_cclp_and_car(ego_snapshots, ego_node, top_k_values, snap_range):

    percent_cclp = {}
    percent_dcaa = {}
    percent_car = {}
    percent_dccn = {}

    for k in top_k_values:
        percent_cclp[k] = []
        percent_dcaa[k] = []
        percent_car[k] = []
        percent_dccn[k] = []

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
            # adding node tuple for adamic adar
            non_edges.append((ego_node, n))

            if n in formed_nodes:
                y_true.append(1)
            else:
                y_true.append(0)

        y_scores_cclp = cclp(ego_snapshots[i], non_edges)
        y_scores_dcaa = degree_corrected_adamic_adar_index(ego_snapshots[i], non_edges, first_hop_nodes)

        y_scores_car = car(ego_snapshots[i], non_edges)
        y_scores_dccn = degree_corrected_common_neighbors_index(ego_snapshots[i], non_edges, first_hop_nodes)

        combo_scores = np.concatenate((np.array(y_scores_cclp).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dcaa).astype(float).reshape(-1, 1),
                                       np.array(y_scores_car).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dccn).astype(float).reshape(-1, 1),
                                       np.array(y_true).reshape(-1, 1)), axis=1)

        combo_scores_cclp_sorted = combo_scores[combo_scores[:, 0].argsort()[::-1]]
        combo_scores_dcaa_sorted = combo_scores[combo_scores[:, 1].argsort()[::-1]]

        combo_scores_car_sorted = combo_scores[combo_scores[:, 2].argsort()[::-1]]
        combo_scores_dccn_sorted = combo_scores[combo_scores[:, 3].argsort()[::-1]]

        for k in percent_cclp.keys():
            percent_cclp[k].append(sum(combo_scores_cclp_sorted[:k, -1]) / k)
            percent_dcaa[k].append(sum(combo_scores_dcaa_sorted[:k, -1]) / k)
            percent_car[k].append(sum(combo_scores_car_sorted[:k, -1]) / k)
            percent_dccn[k].append(sum(combo_scores_dccn_sorted[:k, -1]) / k)

    return percent_cclp, percent_dcaa, percent_car, percent_dccn


def run_adamic_adar_on_ego_net_ranking_only_cclp_and_car(ego_snapshots, ego_node, top_k_values, snap_range):
    percent_cclp = {}
    percent_car = {}

    for k in top_k_values:
        percent_cclp[k] = []
        percent_car[k] = []

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
            # adding node tuple for adamic adar
            non_edges.append((ego_node, n))

            if n in formed_nodes:
                y_true.append(1)
            else:
                y_true.append(0)

        y_scores_cclp = cclp(ego_snapshots[i], non_edges)
        y_scores_car = car(ego_snapshots[i], non_edges)

        combo_scores = np.concatenate((np.array(y_scores_cclp).astype(float).reshape(-1, 1),
                                       np.array(y_scores_car).astype(float).reshape(-1, 1),
                                       np.array(y_true).reshape(-1, 1)), axis=1)

        combo_scores_cclp_sorted = combo_scores[combo_scores[:, 0].argsort()[::-1]]
        combo_scores_car_sorted = combo_scores[combo_scores[:, 1].argsort()[::-1]]

        for k in percent_car.keys():
            percent_cclp[k].append(sum(combo_scores_cclp_sorted[:k, -1]) / k)
            percent_car[k].append(sum(combo_scores_car_sorted[:k, -1]) / k)

    return percent_cclp, percent_car


def run_link_prediction_on_test_method(ego_snapshots, ego_node, top_k_values, snap_range):
    percent_test = {}

    for k in top_k_values:
        percent_test[k] = []

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
            non_edges.append((ego_node, n))

            if n in formed_nodes:
                y_true.append(1)
            else:
                y_true.append(0)

        y_scores_test = test_lp_method(ego_snapshots[i], non_edges, first_hop_nodes)

        combo_scores = np.concatenate((np.array(y_scores_test).astype(float).reshape(-1, 1),
                                       np.array(y_true).reshape(-1, 1)), axis=1)

        combo_scores_test_sorted = combo_scores[combo_scores[:, 0].argsort()[::-1]]

        for k in percent_test.keys():
            percent_test[k].append(sum(combo_scores_test_sorted[:k, -1]) / k)

    return percent_test


def ego_net_link_formation_hop_degree(ego_snapshots, ego_node):
    degree_formation = None
    for i in range(len(ego_snapshots) - 1):
        first_hop_nodes = set(ego_snapshots[i].neighbors(ego_node))

        if len(first_hop_nodes) < 30:
            continue

        second_hop_nodes = set(ego_snapshots[i].nodes()) - first_hop_nodes
        second_hop_nodes.remove(ego_node)

        formed_nodes = second_hop_nodes.intersection(ego_snapshots[i + 1].neighbors(ego_node))

        if len(formed_nodes) == 0 or len(second_hop_nodes) == 0:
            continue

        non_edges = []

        for n in second_hop_nodes:
            # adding node tuple for adamic adar
            non_edges.append((ego_node, n))

        degrees = common_neighbor_hop(ego_snapshots[i], non_edges, first_hop_nodes, formed_nodes)
        if degree_formation is None:
            degree_formation = np.array(degrees)
        else:
            degree_formation = np.concatenate((degree_formation, np.array(degrees)))

    return degree_formation


def run_adamic_adar_on_ego_net_ranking_plot(ego_snapshots, ego_node):
    index_comparison = None
    for i in range(len(ego_snapshots) - 1):
        first_hop_nodes = set(ego_snapshots[i].neighbors(ego_node))

        if len(first_hop_nodes) < 30:
            continue

        second_hop_nodes = set(ego_snapshots[i].nodes()) - first_hop_nodes
        second_hop_nodes.remove(ego_node)

        formed_nodes = second_hop_nodes.intersection(ego_snapshots[i + 1].neighbors(ego_node))

        if len(formed_nodes) == 0 or len(second_hop_nodes) == 0:
            continue

        non_edges = []
        y_true = []

        for n in second_hop_nodes:
            # adding node tuple for adamic adar
            non_edges.append((ego_node, n))

            if n in formed_nodes:
                y_true.append(1)
            else:
                y_true.append(0)

        y_scores_aa = [p for u, v, p in nx.adamic_adar_index(ego_snapshots[i], non_edges)]
        y_scores_aa = np.array(y_scores_aa).astype(float).reshape(1, -1)
        y_scores_aa = list(preprocessing.normalize(y_scores_aa, norm='max')[0])

        y_scores_dcaa = degree_corrected_adamic_adar_index(ego_snapshots[i], non_edges, first_hop_nodes)
        y_scores_dcaa = np.array(y_scores_dcaa).astype(float).reshape(1, -1)
        y_scores_dcaa = list(preprocessing.normalize(y_scores_dcaa, norm='max')[0])

        combo_scores = np.concatenate((np.array(y_scores_aa).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dcaa).astype(float).reshape(-1, 1),
                                       np.array(y_true).reshape(-1, 1)), axis=1)

        if index_comparison is None:
            index_comparison = np.array(combo_scores)
        else:
            index_comparison = np.concatenate((index_comparison, np.array(combo_scores)))

    return index_comparison


##################### Directed Graphs Helpers ########################
def run_link_prediction_comparison_on_directed_graph(ego_net_file, triangle_type, top_k_values):

    triangle_type_func = {
        'T01': dh.get_t01_type_nodes,
        'T02': dh.get_t02_type_nodes,
        'T03': dh.get_t03_type_nodes,
        'T04': dh.get_t04_type_nodes,
        'T05': dh.get_t05_type_nodes,
        'T06': dh.get_t06_type_nodes,
        'T07': dh.get_t07_type_nodes,
        'T08': dh.get_t08_type_nodes,
        'T09': dh.get_t09_type_nodes,
    }

    percent_scores = {
        'cn': {},
        'dccn_i': {},
        'dccn_o': {},
        'aa_i': {},
        'aa_o': {},
        'dcaa_i': {},
        'dcaa_o': {}
    }

    for k in top_k_values:
        for ps in percent_scores.keys():
            percent_scores[ps][k] = []

    with open(ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    tot_num_v_nodes = 0

    # return if the network has less than 30 nodes
    if nx.number_of_nodes(ego_net) < 30 or nx.number_of_nodes(ego_net) > 500000:
        return

    ego_net_snapshots = []

    # find out what snapshot the ego node first appeared in
    first_snapshot = 3
    for u, v, d in ego_net.out_edges(ego_node, data=True):
        if d['snapshot'] < first_snapshot:
            first_snapshot = d['snapshot']
            if first_snapshot == 0:
                break
    if first_snapshot != 0:
        for u, v, d in ego_net.in_edges(ego_node, data=True):
            if d['snapshot'] < first_snapshot:
                first_snapshot = d['snapshot']
                if first_snapshot == 0:
                    break

    if first_snapshot > 2:
        return

    for r in range(first_snapshot, 4):
        temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
        ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes, second_hop_nodes, v_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i], ego_node)

        if len(v_nodes) > 50000 or len(v_nodes) < 30:
            continue

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

        tot_num_v_nodes += len(v_nodes)

        v_nodes_list = list(v_nodes.keys())
        y_true = []

        for v_i in range(0, len(v_nodes_list)):
            if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                y_true.append(1)
            else:
                y_true.append(0)

        y_scores_cn = directed_common_neighbors_index(v_nodes_list, v_nodes)
        y_scores_cn = np.array(y_scores_cn).astype(float).reshape(1, -1)
        y_scores_cn = list(preprocessing.normalize(y_scores_cn, norm='max')[0])

        y_scores_dccn_i = directed_degree_corrected_common_neighbors_index(ego_net, v_nodes_list, v_nodes,
                                                                           first_hop_nodes, True)
        y_scores_dccn_i = np.array(y_scores_dccn_i).astype(float).reshape(1, -1)
        y_scores_dccn_i = list(preprocessing.normalize(y_scores_dccn_i, norm='max')[0])

        y_scores_dccn_o = directed_degree_corrected_common_neighbors_index(ego_net, v_nodes_list, v_nodes,
                                                                           first_hop_nodes, False)
        y_scores_dccn_o = np.array(y_scores_dccn_o).astype(float).reshape(1, -1)
        y_scores_dccn_o = list(preprocessing.normalize(y_scores_dccn_o, norm='max')[0])

        y_scores_aa_i = directed_adamic_adar_index(ego_net, v_nodes_list, v_nodes, True)
        y_scores_aa_i = np.array(y_scores_aa_i).astype(float).reshape(1, -1)
        y_scores_aa_i = list(preprocessing.normalize(y_scores_aa_i, norm='max')[0])

        y_scores_aa_o = directed_adamic_adar_index(ego_net, v_nodes_list, v_nodes, False)
        y_scores_aa_o = np.array(y_scores_aa_o).astype(float).reshape(1, -1)
        y_scores_aa_o = list(preprocessing.normalize(y_scores_aa_o, norm='max')[0])

        y_scores_dcaa_i = directed_degree_corrected_adamic_adar_index(ego_net, v_nodes_list, v_nodes, first_hop_nodes,
                                                                      True)
        y_scores_dcaa_i = np.array(y_scores_dcaa_i).astype(float).reshape(1, -1)
        y_scores_dcaa_i = list(preprocessing.normalize(y_scores_dcaa_i, norm='max')[0])

        y_scores_dcaa_o = directed_degree_corrected_adamic_adar_index(ego_net, v_nodes_list, v_nodes, first_hop_nodes,
                                                                      True)
        y_scores_dcaa_o = np.array(y_scores_dcaa_o).astype(float).reshape(1, -1)
        y_scores_dcaa_o = list(preprocessing.normalize(y_scores_dcaa_o, norm='max')[0])

        combo_scores = np.concatenate((np.array(y_scores_cn).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dccn_i).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dccn_o).astype(float).reshape(-1, 1),
                                       np.array(y_scores_aa_i).astype(float).reshape(-1, 1),
                                       np.array(y_scores_aa_o).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dcaa_i).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dcaa_o).astype(float).reshape(-1, 1),
                                       np.array(y_true).reshape(-1, 1)), axis=1)

        combo_scores_cn_sorted = combo_scores[combo_scores[:, 0].argsort()[::-1]]
        combo_scores_dccn_i_sorted = combo_scores[combo_scores[:, 1].argsort()[::-1]]
        combo_scores_dccn_o_sorted = combo_scores[combo_scores[:, 2].argsort()[::-1]]
        combo_scores_aa_i_sorted = combo_scores[combo_scores[:, 3].argsort()[::-1]]
        combo_scores_aa_o_sorted = combo_scores[combo_scores[:, 4].argsort()[::-1]]
        combo_scores_dcaa_i_sorted = combo_scores[combo_scores[:, 5].argsort()[::-1]]
        combo_scores_dcaa_o_sorted = combo_scores[combo_scores[:, 6].argsort()[::-1]]

        for k in top_k_values:
            percent_scores['cn'][k].append(sum(combo_scores_cn_sorted[:k, -1]) / k)
            percent_scores['dccn_i'][k].append(sum(combo_scores_dccn_i_sorted[:k, -1]) / k)
            percent_scores['dccn_o'][k].append(sum(combo_scores_dccn_o_sorted[:k, -1]) / k)
            percent_scores['aa_i'][k].append(sum(combo_scores_aa_i_sorted[:k, -1]) / k)
            percent_scores['aa_o'][k].append(sum(combo_scores_aa_o_sorted[:k, -1]) / k)
            percent_scores['dcaa_i'][k].append(sum(combo_scores_dcaa_i_sorted[:k, -1]) / k)
            percent_scores['dcaa_o'][k].append(sum(combo_scores_dcaa_o_sorted[:k, -1]) / k)

    return percent_scores


def run_link_prediction_comparison_on_directed_graph_all_types(ego_net_file, top_k_values):
    data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/'
    result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/'

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    triangle_type_func = {
        'T01': dh.get_t01_type_nodes,
        'T02': dh.get_t02_type_nodes,
        'T03': dh.get_t03_type_nodes,
        'T04': dh.get_t04_type_nodes,
        'T05': dh.get_t05_type_nodes,
        'T06': dh.get_t06_type_nodes,
        'T07': dh.get_t07_type_nodes,
        'T08': dh.get_t08_type_nodes,
        'T09': dh.get_t09_type_nodes,
    }

    score_list = ['cn', 'dccn_i', 'dccn_o', 'aa_i', 'aa_o', 'dcaa_i', 'dcaa_o']

    for triangle_type in triangle_type_func.keys():

        percent_scores = {
            'cn': {},
            'dccn_i': {},
            'dccn_o': {},
            'aa_i': {},
            'aa_o': {},
            'dcaa_i': {},
            'dcaa_o': {}
        }

        for k in top_k_values:
            for ps in percent_scores.keys():
                percent_scores[ps][k] = []

        with open(data_file_base_path + ego_net_file, 'rb') as f:
            ego_node, ego_net = pickle.load(f)

        ego_net_snapshots = []

        # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
        if nx.number_of_nodes(ego_net) > 50000:
            with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
                pickle.dump(0, f, protocol=-1)

        for r in range(0, 4):
            temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
            ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

        # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
        for i in range(len(ego_net_snapshots) - 1):
            first_hop_nodes, second_hop_nodes, v_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i], ego_node)

            if len(first_hop_nodes) < 10:
                continue

            v_nodes_list = list(v_nodes.keys())
            y_true = []

            for v_i in range(0, len(v_nodes_list)):
                if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                    y_true.append(1)
                else:
                    y_true.append(0)

            # continue of no edge was formed
            if np.sum(y_true) == 0:
                continue

            lp_scores = all_directed_lp_indices(ego_net, v_nodes_list, v_nodes, first_hop_nodes)

            lp_scores = np.concatenate((lp_scores, np.array(y_true).reshape(-1, 1)), axis=1)

            for si in range(len(score_list)):
                lp_score_sorted = lp_scores[lp_scores[:, si].argsort()[::-1]]
                for k in top_k_values:
                    percent_scores[score_list[si]][k].append(sum(lp_score_sorted[:k, -1]) / k)

        # skip if no snapshot returned a score
        if len(percent_scores[score_list[0]][top_k_values[0]]) == 0:
            continue

        # getting the mean of all snapshots for each score
        for s in percent_scores:
            for k in top_k_values:
                percent_scores[s][k] = np.mean(percent_scores[s][k])

        with open(result_file_base_path + triangle_type + '/' + ego_net_file, 'wb') as f:
            pickle.dump(percent_scores, f, protocol=-1)

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    print("Analyzed ego net: " + ego_net_file)


def run_link_prediction_comparison_on_directed_graph_all_types_based_on_empirical(ego_net_file, top_k_values):
    data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/'
    result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/'
    empirical_analyzed_egonets_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files/'

    triangle_type_func = {
        'T01': dh.get_t01_type_nodes,
        'T02': dh.get_t02_type_nodes,
        'T03': dh.get_t03_type_nodes,
        'T04': dh.get_t04_type_nodes,
        'T05': dh.get_t05_type_nodes,
        'T06': dh.get_t06_type_nodes,
        'T07': dh.get_t07_type_nodes,
        'T08': dh.get_t08_type_nodes,
        'T09': dh.get_t09_type_nodes,
    }

    triangle_types_to_analyze = []
    for tt in triangle_type_func:
        if not os.path.isfile(result_file_base_path + tt + '/' + ego_net_file) and \
               os.path.isfile(empirical_analyzed_egonets_path + tt + '/' + ego_net_file):
            triangle_types_to_analyze.append(tt)

    if len(triangle_types_to_analyze) == 0:
        return

    score_list = ['cn', 'dccn_i', 'dccn_o', 'aa_i', 'aa_o', 'dcaa_i', 'dcaa_o']

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    ego_net_snapshots = []

    for r in range(0, 4):
        temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
        ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

    for triangle_type in triangle_types_to_analyze:

        percent_scores = {
            'cn': {},
            'dccn_i': {},
            'dccn_o': {},
            'aa_i': {},
            'aa_o': {},
            'dcaa_i': {},
            'dcaa_o': {}
        }

        for k in top_k_values:
            for ps in percent_scores.keys():
                percent_scores[ps][k] = []

        # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
        for i in range(len(ego_net_snapshots) - 1):
            first_hop_nodes, second_hop_nodes, v_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i],
                                                                                           ego_node)

            if len(first_hop_nodes) < 10:
                continue

            v_nodes_list = list(v_nodes.keys())
            y_true = []

            for v_i in range(0, len(v_nodes_list)):
                if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                    y_true.append(1)
                else:
                    y_true.append(0)

            # continue of no edge was formed
            if np.sum(y_true) == 0:
                continue

            lp_scores = all_directed_lp_indices(ego_net, v_nodes_list, v_nodes, first_hop_nodes)

            lp_scores = np.concatenate((lp_scores, np.array(y_true).reshape(-1, 1)), axis=1)

            for si in range(len(score_list)):
                lp_score_sorted = lp_scores[lp_scores[:, si].argsort()[::-1]]
                for k in top_k_values:
                    percent_scores[score_list[si]][k].append(sum(lp_score_sorted[:k, -1]) / k)

        # skip if no snapshot returned a score
        if len(percent_scores[score_list[0]][top_k_values[0]]) == 0:
            continue

        # getting the mean of all snapshots for each score
        for s in percent_scores:
            for k in top_k_values:
                percent_scores[s][k] = np.mean(percent_scores[s][k])

        with open(result_file_base_path + triangle_type + '/' + ego_net_file, 'wb') as f:
            pickle.dump(percent_scores, f, protocol=-1)

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    print("Analyzed ego net: " + ego_net_file)


def all_directed_lp_indices(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):
    # every row is a v node, and every column is a score in the following order:
    # cn, dccn_i, dccn_o, aa_i, aa_o, dcaa_i, dcaa_o

    scores = np.zeros((len(v_nodes_list), 7))

    for v_i in range(0, len(v_nodes_list)):
        # cn score
        scores[v_i, 0] = len(v_nodes_z[v_nodes_list[v_i]])

        temp_dccn_i_score = 0
        temp_dccn_o_score = 0

        temp_aa_i_score = 0
        temp_aa_o_score = 0

        temp_dcaa_i_score = 0
        temp_dcaa_o_score = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:

            z_pred = set(ego_net.predecessors(z))
            z_succ = set(ego_net.successors(z))

            # shift up all degrees by 2 so if it 1 or less we will not encounter an error
            z_global_indegree = len(z_pred) + 1
            z_global_outdegree = len(z_succ) + 1

            z_local_indegree = len(z_pred.intersection(first_hop_nodes)) + 1
            z_local_outdegree = len(z_succ.intersection(first_hop_nodes)) + 1

            y_indegree = z_global_indegree - z_local_indegree
            y_outdegree = z_global_outdegree - z_local_outdegree

            temp_dccn_i_score += math.log(z_local_indegree + 1)
            temp_dccn_o_score += math.log(z_local_outdegree + 1)

            temp_aa_i_score += 1 / math.log(z_global_indegree + 1)
            temp_aa_o_score += 1 / math.log(z_global_outdegree + 1)

            temp_dcaa_i_score += 1 / math.log((z_local_indegree * (1 - (z_local_indegree / z_global_indegree))) +
                                              (y_indegree * (z_global_indegree / z_local_indegree)) + 2)
            temp_dcaa_o_score += 1 / math.log((z_local_outdegree * (1 - (z_local_outdegree / z_global_outdegree))) +
                                              (y_outdegree * (z_global_outdegree / z_local_outdegree)) + 2)

        # dccn_i score
        scores[v_i, 1] = temp_dccn_i_score
        # dccn_o degree
        scores[v_i, 2] = temp_dccn_o_score
        # aa_i score
        scores[v_i, 3] = temp_aa_i_score
        # aa_o degree
        scores[v_i, 4] = temp_aa_o_score
        # dcaa_i score
        scores[v_i, 5] = temp_dcaa_i_score
        # dcaa_o degree
        scores[v_i, 6] = temp_dcaa_o_score

    return scores


def directed_common_neighbors_index(v_nodes_list, v_nodes_z):
    scores = []

    for v_i in range(0, len(v_nodes_list)):
        scores.append(len(v_nodes_z[v_nodes_list[v_i]]))

    return scores


def directed_degree_corrected_common_neighbors_index(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes, in_degree=True):
    scores = []

    for v_i in range(0, len(v_nodes_list)):
        temp_score = 0
        for z in v_nodes_z[v_nodes_list[v_i]]:
            d = 0
            if in_degree:
                d = len(set(ego_net.predecessors(z)).intersection(first_hop_nodes))
            else:
                d = len(set(ego_net.successors(z)).intersection(first_hop_nodes))

            temp_score += math.log(d + 2)

        scores.append(temp_score)
    return scores


def directed_adamic_adar_index(ego_net, v_nodes_list, v_nodes_z, in_degree=True):
    scores = []

    for v_i in range(0, len(v_nodes_list)):
        temp_score = 0
        for z in v_nodes_z[v_nodes_list[v_i]]:
            d = 0
            if in_degree:
                d = len(set(ego_net.predecessors(z)))
            else:
                d = len(set(ego_net.successors(z)))

            if d > 2:
                temp_score += (1 / math.log(d))

        scores.append(temp_score)
    return scores


def directed_degree_corrected_adamic_adar_index(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes, in_degree=True):
    scores = []

    for v_i in range(0, len(v_nodes_list)):
        temp_score = 0
        for z in v_nodes_z[v_nodes_list[v_i]]:

            if in_degree:
                z_ps = set(ego_net.predecessors(z))
            else:
                z_ps = set(ego_net.successors(z))

            # total degree
            t = len(z_ps)

            # local degree
            x = len(z_ps.intersection(first_hop_nodes))

            # total degree - local degree
            y = t - x

            if x == 0 or t == 0:
                continue

            temp_score += (1 / math.log((x * (1 - x / t)) + (y * (t / x))))

        scores.append(temp_score)
    return scores


###### Directed LP on combined triads #######
def run_link_prediction_comparison_on_directed_graph_combined_types(ego_net_file, top_k_values):
    data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/'
    # result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/'
    result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/test-2/'

    startTime = datetime.now()

    # return if the egonet is on the analyzed list
    # if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
    #     return

    # return if the egonet is on the skipped list
    # if os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
    #     return

    score_list = ['cn', 'dccn', 'aa', 'dcaa']

    percent_scores = {
        'cn': {},
        'dccn': {},
        'aa': {},
        'dcaa': {}
    }

    for k in top_k_values:
        for ps in percent_scores.keys():
            percent_scores[ps][k] = []

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    ego_net_snapshots = []
    total_y_true = 0

    num_nodes = nx.number_of_nodes(ego_net)
    # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
    if num_nodes >= 10:
        # with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
        #     pickle.dump(0, f, protocol=-1)
        return

    os.remove(result_file_base_path + 'skipped_egonets/' + ego_net_file)
    # if nx.number_of_nodes(ego_net) <= 25000:
    #     return

    for r in range(0, 4):
        temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
        ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes, second_hop_nodes, v_nodes = dh.get_combined_type_nodes(ego_net_snapshots[i], ego_node)

        # if len(first_hop_nodes) < 10:
        #     continue

        v_nodes_list = list(v_nodes.keys())
        y_true = []

        for v_i in range(0, len(v_nodes_list)):
            if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                y_true.append(1)
            else:
                y_true.append(0)

        # continue of no edge was formed
        if np.sum(y_true) == 0:
            continue

        total_y_true += np.sum(y_true)

        lp_scores = all_directed_lp_indices_combined(ego_net, v_nodes_list, v_nodes, first_hop_nodes)

        lp_scores = np.concatenate((lp_scores, np.array(y_true).reshape(-1, 1)), axis=1)

        for si in range(len(score_list)):
            lp_score_sorted = lp_scores[lp_scores[:, si].argsort()[::-1]]
            for k in top_k_values:
                percent_scores[score_list[si]][k].append(sum(lp_score_sorted[:k, -1]) / k)

    # skip if no snapshot returned a score
    if len(percent_scores[score_list[0]][top_k_values[0]]) > 0:
        # getting the mean of all snapshots for each score
        for s in percent_scores:
            for k in top_k_values:
                percent_scores[s][k] = np.mean(percent_scores[s][k])

        with open(result_file_base_path + 'results/' + ego_net_file, 'wb') as f:
            pickle.dump(percent_scores, f, protocol=-1)

        print("Analyzed ego net: {0} - Duration: {1} - Num nodes: {2} - Formed: {3}"
              .format(ego_net_file, datetime.now() - startTime, nx.number_of_nodes(ego_net), total_y_true))

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)


def all_directed_lp_indices_combined(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):
    # every row is a v node, and every column is a score in the following order:
    # cn, dccn, aa, dcaa

    scores = np.zeros((len(v_nodes_list), 4))

    for v_i in range(0, len(v_nodes_list)):
        # cn score
        scores[v_i, 0] = len(v_nodes_z[v_nodes_list[v_i]])

        temp_dccn = 0
        temp_aa = 0
        temp_dcaa = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            z_neighbors = set(ego_net.predecessors(z)).union(set(ego_net.successors(z)))

            z_global_degree = len(z_neighbors)

            z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

            y = z_global_degree - z_local_degree

            # temp_dccn += math.log(z_local_degree + 2)
            temp_dccn += (z_local_degree + len(v_nodes_z[v_nodes_list[v_i]]))

            temp_aa += 1 / math.log(z_global_degree)

            z_local_degree += 1
            temp_dcaa += 1 / math.log((z_local_degree * (1 - (z_local_degree / z_global_degree))) +
                                      (y * (z_global_degree / z_local_degree)))

        # dccn score
        scores[v_i, 1] = temp_dccn
        # aa degree
        scores[v_i, 2] = temp_aa
        # dcaa score
        scores[v_i, 3] = temp_dcaa

    return scores


def run_link_prediction_comparison_on_directed_graph_combined_types_only_car_and_cclp(ego_net_file, top_k_values):
    data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/'
    result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/' \
                            'test-3-car-cclp/'

    startTime = datetime.now()

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    score_list = ['car', 'cclp']

    percent_scores = {
        'car': {},
        'cclp': {}
    }

    for k in top_k_values:
        for ps in percent_scores.keys():
            percent_scores[ps][k] = []

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    ego_net_snapshots = []
    total_y_true = 0

    # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
    if nx.number_of_nodes(ego_net) > 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)
        return

    # os.remove(result_file_base_path + 'skipped_egonets/' + ego_net_file)
    # if nx.number_of_nodes(ego_net) <= 25000:
    #     return

    for r in range(0, 4):
        temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
        ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes, second_hop_nodes, v_nodes = dh.get_combined_type_nodes(ego_net_snapshots[i], ego_node)

        v_nodes_list = list(v_nodes.keys())
        y_true = []

        for v_i in range(0, len(v_nodes_list)):
            if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                y_true.append(1)
            else:
                y_true.append(0)

        # continue of no edge was formed
        if np.sum(y_true) == 0:
            continue

        total_y_true += np.sum(y_true)

        lp_scores = car_and_cclp_directed_lp_indices_combined(ego_net, v_nodes_list, v_nodes)

        lp_scores = np.concatenate((lp_scores, np.array(y_true).reshape(-1, 1)), axis=1)

        for si in range(len(score_list)):
            lp_score_sorted = lp_scores[lp_scores[:, si].argsort()[::-1]]
            for k in top_k_values:
                percent_scores[score_list[si]][k].append(sum(lp_score_sorted[:k, -1]) / k)

    # skip if no snapshot returned a score
    if len(percent_scores[score_list[0]][top_k_values[0]]) > 0:
        # getting the mean of all snapshots for each score
        for s in percent_scores:
            for k in top_k_values:
                percent_scores[s][k] = np.mean(percent_scores[s][k])

        with open(result_file_base_path + 'results/' + ego_net_file, 'wb') as f:
            pickle.dump(percent_scores, f, protocol=-1)

        print("Analyzed ego net: {0} - Duration: {1} - Num nodes: {2} - Formed: {3}"
              .format(ego_net_file, datetime.now() - startTime, nx.number_of_nodes(ego_net), total_y_true))

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)


def car_and_cclp_directed_lp_indices_combined(ego_net, v_nodes_list, v_nodes_z):
    # every row is a v node, and every column is a score in the following order:
    # car, cclp

    scores = np.zeros((len(v_nodes_list), 2))
    undirected_ego_net = ego_net.to_undirected()

    for v_i in range(0, len(v_nodes_list)):
        num_cn = len(v_nodes_z[v_nodes_list[v_i]])
        lcl = undirected_ego_net.subgraph(v_nodes_z[v_nodes_list[v_i]]).number_of_edges()

        # car score
        scores[v_i, 0] = num_cn * lcl

        temp_cclp = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            z_deg = undirected_ego_net.degree(z)
            z_tri = nx.triangles(undirected_ego_net, z)

            temp_cclp += z_tri / (z_deg * (z_deg - 1) / 2)

        # cclp score
        scores[v_i, 1] = temp_cclp

    return scores
