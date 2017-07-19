import networkx as nx
from sklearn import metrics


def run_adamic_adar_on_ego_net(ego_snapshots, ego_node):
    lp_result = []
    for i in range(len(ego_snapshots) - 1):
        second_hop_nodes = set(ego_snapshots[i].nodes()) - set(ego_snapshots[i].neighbors(ego_node))
        second_hop_nodes.remove(ego_node)

        formed_nodes = second_hop_nodes.intersection(ego_snapshots[i + 1].neighbors(ego_node))
        non_edges = []
        y_true = []

        for n in second_hop_nodes:
            # adding node tuple for adamic adar
            non_edges.append((ego_node, n))

            if n in formed_nodes:
                y_true.append(1)
            else:
                y_true.append(0)

        y_scores = nx.adamic_adar_index(ego_snapshots[i], non_edges)

        lp_result.append(evaluate_prediction_result(y_true, y_scores))

    return lp_result


def evaluate_prediction_result(y_true, y_scores):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    average_precision = metrics.average_precision_score(y_true, y_scores)

    result = {
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'auroc': auroc,
        'average_precision': average_precision
    }

    return result
