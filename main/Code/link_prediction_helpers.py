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


def degree_corrected_adamic_adar_index(ego_net, non_edges, first_hop_nodes):
    scores = []

    for u, v in non_edges:
        first_hop_degrees = []
        other_degrees = []
        common_neighbors = nx.common_neighbors(ego_net, u, v)

        for c in common_neighbors:
            cn_neighbors = set(nx.neighbors(ego_net, c))
            first_hop_degrees.append(len(cn_neighbors.intersection(first_hop_nodes)))
            other_degrees.append(len(cn_neighbors - first_hop_nodes))

        scores.append(sum(1 / math.log(d) for d in other_degrees))

    return scores


def plot_auroc_hist(lp_results):
    aurocs = {}
    for lp_method in lp_results[0]:
        aurocs[lp_method] = []

    for lp_result in lp_results:
        for lp_method in lp_result:
            aurocs[lp_method].append(lp_result[lp_method]['auroc'])

    plt.figure()
    bins = np.linspace(0, 1, 100)
    for lp_method in aurocs:
        plt.hist(aurocs[lp_method], bins, label='%s' % lp_method, alpha=0.6)
    plt.ylabel('Frequency')
    plt.xlabel('Area Under ROC')
    plt.title('AUROC Histogram of Ego Centric Graphs')
    plt.legend(loc="upper left")
    plt.show()
