import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
        }
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

        y_scores_aa = [p for u, v, p in nx.adamic_adar_index(ego_snapshots[i], non_edges)]
        y_scores_aa = np.array(y_scores_aa).astype(float).reshape(1, -1)
        y_scores_aa = list(preprocessing.normalize(y_scores_aa, norm='max')[0])
        evaluate_prediction_result(y_true, y_scores_aa, lp_result['adamic_adar'])

        y_scores_dcaa = degree_corrected_adamic_adar_index(ego_snapshots[i], non_edges, first_hop_nodes)
        # y_scores_dcaa = shift_up(y_scores_dcaa)
        y_scores_dcaa = np.array(y_scores_dcaa).astype(float).reshape(1, -1)
        y_scores_dcaa = list(preprocessing.normalize(y_scores_dcaa, norm='max')[0])
        evaluate_prediction_result(y_true, y_scores_dcaa, lp_result['degree_corrected_adamic_adar'])

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


def degree_corrected_adamic_adar_index(ego_net, non_edges, first_hop_nodes, orig_snap):
    scores = []

    for u, v in non_edges:
        # first_hop_degrees = []
        other_degrees = []
        common_neighbors = nx.common_neighbors(orig_snap, u, v)

        cn = 0
        for c in common_neighbors:
            # cn_neighbors = set(nx.neighbors(orig_snap, c))
            cn += len(list(nx.common_neighbors(orig_snap, u, c)))
            # cn += len(list(nx.common_neighbors(orig_snap, v, c)))
            # x = len(cn_neighbors.intersection(first_hop_nodes))
            # # first_hop_degrees.append(x ** 2 / (len(cn_neighbors) * len(first_hop_nodes)))
            # first_hop_degrees.append(x)
            # other_degrees.append(len(cn_neighbors))

        # for i in range(len(first_hop_degrees)):
        #     if first_hop_degrees[i] == 0:
        #         first_hop_degrees[i] = 1.33
        #     elif first_hop_degrees[i] == 1:
        #         first_hop_degrees[i] = 1.66

        # other_degrees_index = sum((math.log(d) * -1) for d in other_degrees)
        # first_hop_degree_index = sum(math.log(d) for d in other_degrees)
        # first_hop_degree_index = sum(first_hop_degrees)
        scores.append(cn)

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


def run_adamic_adar_on_ego_net_ranking(ego_snapshots, ego_node, orig_snaps):
    percent_aa = []
    percent_dcaa = []
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

        y_scores_dcaa = degree_corrected_adamic_adar_index(ego_snapshots[i], non_edges, first_hop_nodes, orig_snaps[i])

        combo_scores = np.concatenate((np.array(y_scores_aa).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dcaa).astype(float).reshape(-1, 1),
                                       np.array(y_true).reshape(-1, 1)), axis=1)

        combo_scores_aa_sorted = combo_scores[combo_scores[:, 0].argsort()[::-1]]
        combo_scores_dcaa_sorted = combo_scores[combo_scores[:, 1].argsort()[::-1]]
        ones_index_aa = np.where(combo_scores_aa_sorted[:, 2] == 1)[0]
        ones_index_dcaa = np.where(combo_scores_dcaa_sorted[:, 2] == 1)[0]
        ones_aa = combo_scores_aa_sorted[ones_index_aa]
        ones_dcaa = combo_scores_dcaa_sorted[ones_index_dcaa]

        # top_n = math.ceil(len(y_true) * 0.03)
        # percent_aa.append(sum(combo_scores_aa_sorted[:top_n, 2]) / top_n)
        # percent_dcaa.append(sum(combo_scores_dcaa_sorted[:top_n, 2]) / top_n)

        ones_index_aa = ones_index_aa / len(y_true)
        ones_index_dcaa = ones_index_dcaa / len(y_true)

        for m in ones_index_aa:
            percent_aa.append(m)

        for m in ones_index_dcaa:
            percent_dcaa.append(m)

    return percent_aa, percent_dcaa


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