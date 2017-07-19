import re
import math
import random
import numpy as np
import networkx as nx
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm


def run_link_prediction(graph):
    # get all the non existing edges before deleting random existing ones
    non_existing_edges = list(nx.non_edges(graph))

    # delete random existing edges
    deleted_edges = delete_edges_randomly(graph, 10)

    # Create a y_score array
    ones_array = np.ones((len(deleted_edges),), dtype=np.int)
    zeros_array = np.zeros((len(non_existing_edges),), dtype=np.int)
    y_true = np.concatenate((zeros_array, ones_array), axis=0)

    lp_results = {}

    lp_results["adamic_adar"] = get_prediction_result(y_true,
            get_y_score(nx.adamic_adar_index(graph, non_existing_edges),
                        nx.adamic_adar_index(graph, deleted_edges)))

    lp_results["preferential_attachment"] = get_prediction_result(y_true,
            get_y_score(nx.preferential_attachment(graph, non_existing_edges),
                        nx.preferential_attachment(graph, deleted_edges)))

    lp_results["jaccard_coefficient"] = get_prediction_result(y_true,
            get_y_score(nx.jaccard_coefficient(graph, non_existing_edges),
                        nx.jaccard_coefficient(graph, deleted_edges)))

    lp_results["resource_allocation_index"] = get_prediction_result(y_true,
            get_y_score(nx.resource_allocation_index(graph, non_existing_edges),
                        nx.resource_allocation_index(graph, deleted_edges)))

    return lp_results


def delete_edges_randomly(graph, percent):
    num_of_edges = nx.number_of_edges(graph)
    percent_num_of_edges = math.ceil(num_of_edges / percent)
    deleted_edges = []

    for num in range(0, percent_num_of_edges):
        existing_edges = nx.edges(graph)
        random_num = random.randrange(len(existing_edges))
        deleted_edges.append(existing_edges[random_num])
        target_edge = existing_edges[random_num]
        graph.remove_edge(*target_edge)

    return deleted_edges


def get_y_score(non_existing_edges, deleted_edges):
    deleted_edges_score = []
    non_existing_edges_score = []

    for u, v, p in deleted_edges:
        deleted_edges_score.append(p)

    for u, v, p in non_existing_edges:
        non_existing_edges_score.append(p)

    y_scores = np.array(non_existing_edges_score + deleted_edges_score)

    return y_scores


def get_prediction_result(y_true, y_scores):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    # precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    average_precision = metrics.average_precision_score(y_true, y_scores)

    result = {
        # 'fpr': fpr,
        # 'tpr': tpr,
        # 'precision': precision,
        # 'recall': recall,
        'auroc': auroc,
        'average_precision': average_precision
    }

    return result


def plot_roc(lp_result):
    plt.figure()
    for lp_method in lp_result:
        plt.plot(lp_result[lp_method]['fpr'], lp_result[lp_method]['tpr'],
                 label='%s (AUC = %0.2f)' % (lp_method, lp_result[lp_method]['auroc']))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_pr(lp_result):
    plt.clf()
    for lp_method in lp_result:
        plt.plot(lp_result[lp_method]['recall'], lp_result[lp_method]['precision'],
                 label='%s (AUC = %0.2f)' % (lp_method, lp_result[lp_method]['average_precision']))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall Curve')
    plt.legend(loc="upper right")
    plt.show()


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
        plt.hist(aurocs[lp_method], bins, label='%s' % lp_method)
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

    plt.figure()
    bins = np.linspace(0, 1, 100)
    for lp_method in aupr:
        plt.hist(aupr[lp_method], bins, label='%s' % lp_method)
    plt.ylabel('Frequency')
    plt.xlabel('Area Under Precision Recall (Average Precision)')
    plt.title('AUPR Histogram of Ego Centric Graphs')
    plt.legend(loc="upper right")
    plt.show()


def plot_num_of_edge_node_hist(graphs):
    num_of_edges = []
    num_of_nodes = []
    for graph in graphs:
        num_of_edges.append(nx.number_of_edges(graph))
        num_of_nodes.append(nx.number_of_nodes(graph))

    plt.figure()
    bins = np.linspace(0, max(num_of_edges), 100)
    plt.hist(num_of_edges, bins, label="Frequency of number of edges")
    plt.ylabel('Frequency')
    plt.xlabel('Number of Edges')
    plt.title('Histogram of Number of Edges in Ego Centric Graphs')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure()
    bins = np.linspace(0, max(num_of_nodes), 100)
    plt.hist(num_of_nodes, bins, label="Frequency of number of nodes")
    plt.ylabel('Frequency')
    plt.xlabel('Number of Nodes')
    plt.title('Histogram of Number of Nodes in Ego Centric Graphs')
    plt.legend(loc="upper right")
    plt.show()


def plot_cdf(graphs):
    num_of_edges = []
    num_of_nodes = []
    for graph in graphs:
        num_of_edges.append(nx.number_of_edges(graph))
        num_of_nodes.append(nx.number_of_nodes(graph))

    ecdf = sm.distributions.ECDF(num_of_nodes)

    plt.figure()
    x = np.linspace(min(num_of_nodes), max(num_of_nodes))
    y = ecdf(x)
    plt.step(x, y)
    plt.ylabel("Probability")
    plt.xlabel("Number of Nodes")
    plt.title("ECDF of Number of Nodes in All Ego Centric Networks")
    plt.show()

    ecdf = sm.distributions.ECDF(num_of_edges)

    plt.figure()
    x = np.linspace(min(num_of_edges), max(num_of_edges))
    y = ecdf(x)
    plt.step(x, y)
    plt.ylabel("Probability")
    plt.xlabel("Number of Edges")
    plt.title("ECDF of Number of Edges in All Ego Centric Networks")
    plt.show()


def explode_to_ego_centric(graph):
    ego_centric_graphs = []

    cnt = 0
    for node in nx.nodes(graph):
        ego_centric_graphs.append(nx.ego_graph(graph, node))
        cnt +=1
        if cnt > 5:
            break

    return ego_centric_graphs


def read_facebook_graph(file_path):
    file = open(file_path, 'r')

    graph = nx.Graph()

    for l in file:
        p = re.compile('\d+')
        nums = p.findall(l)

        nums[0] = int(nums[0])
        nums[1] = int(nums[1])

        if not graph.has_node(nums[0]):
            graph.add_node(nums[0])

        if not graph.has_node(nums[1]):
            graph.add_node(nums[1])

        if len(nums) == 2:
            nums.append(-1)
        else:
            nums[2] = int(nums[2])

        graph.add_edge(nums[0], nums[1], timestamp=nums[2])

    return graph
