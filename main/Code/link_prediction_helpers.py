import math
import pickle
import numpy as np
import networkx as nx
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


def run_adamic_adar_on_ego_net_ranking(ego_snapshots, ego_node, top_k_values):

    percent_aa = {}
    percent_dcaa = {}

    for k in top_k_values:
        percent_aa[k] = []
        percent_dcaa[k] = []

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

        # y_scores_aa = [p for u, v, p in nx.adamic_adar_index(ego_snapshots[i], non_edges)]
        y_scores_aa = common_neighbors_index(ego_snapshots[i], non_edges)

        # y_scores_dcaa = degree_corrected_adamic_adar_index(ego_snapshots[i], non_edges, first_hop_nodes)
        y_scores_dcaa = degree_corrected_common_neighbors_index(ego_snapshots[i], non_edges, first_hop_nodes)

        combo_scores = np.concatenate((np.array(y_scores_aa).astype(float).reshape(-1, 1),
                                       np.array(y_scores_dcaa).astype(float).reshape(-1, 1),
                                       np.array(y_true).reshape(-1, 1)), axis=1)

        combo_scores_aa_sorted = combo_scores[combo_scores[:, 0].argsort()[::-1]]
        combo_scores_dcaa_sorted = combo_scores[combo_scores[:, 1].argsort()[::-1]]
        # ones_index_aa = np.where(combo_scores_aa_sorted[:, 2] == 1)[0]
        # ones_index_dcaa = np.where(combo_scores_dcaa_sorted[:, 2] == 1)[0]
        # ones_aa = combo_scores_aa_sorted[ones_index_aa]
        # ones_dcaa = combo_scores_dcaa_sorted[ones_index_dcaa]

        # top_n = math.ceil(len(y_true) * 0.03)
        for k in percent_aa.keys():
            percent_aa[k].append(sum(combo_scores_aa_sorted[:k, -1]) / k)
            percent_dcaa[k].append(sum(combo_scores_dcaa_sorted[:k, -1]) / k)

        # ones_index_aa = ones_index_aa / len(y_true)
        # ones_index_dcaa = ones_index_dcaa / len(y_true)
        #
        # for m in ones_index_aa:
        #     percent_aa.append(m)
        #
        # for m in ones_index_dcaa:
        #     percent_dcaa.append(m)

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


##################### Directed Graphs Helpers ########################
def run_link_prediction_comparison_on_directed_graph(ego_net_file, triangle_type):
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

    with open(ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    tot_num_v_nodes = 0

    # return if the network has less than 30 nodes
    if nx.number_of_nodes(ego_net) < 30:
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

        # ANALYSIS
        formed_z_in_degree_first_hop = []
        formed_z_out_degree_first_hop = []

        not_formed_z_in_degree_first_hop = []
        not_formed_z_out_degree_first_hop = []

        for v in v_nodes_list:
            if ego_net_snapshots[i + 1].has_edge(ego_node, v):
                y_true.append(1)
            else:
                y_true.append(0)

        for v in v_nodes:
            temp_in_degree_first_hop = []
            temp_out_degree_first_hop = []
            temp_in_degree_second_hop = []
            temp_out_degree_second_hop = []

            for z in v_nodes[v]:
                z_preds = set(ego_net_snapshots[i].predecessors(z))
                z_succs = set(ego_net_snapshots[i].successors(z))

                z_local_in_degree = len(z_preds.intersection(first_hop_nodes))
                z_local_out_degree = len(z_preds.intersection(second_hop_nodes))


                temp_in_degree_first_hop.append()
                temp_in_degree_second_hop.append()
                temp_out_degree_first_hop.append(len(z_succs.intersection(first_hop_nodes)))
                temp_out_degree_second_hop.append(len(z_succs.intersection(second_hop_nodes)))

            if ego_net_snapshots[i + 1].has_edge(ego_node, v):
                formed_z_in_degree_first_hop.append(np.mean(temp_in_degree_first_hop))
                formed_z_out_degree_first_hop.append(np.mean(temp_out_degree_first_hop))

            else:
                not_formed_z_in_degree_first_hop.append(np.mean(temp_in_degree_first_hop))
                not_formed_z_out_degree_first_hop.append(np.mean(temp_out_degree_first_hop))

