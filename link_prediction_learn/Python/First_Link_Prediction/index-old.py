import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import link_prediction_helpers as helpers
from sklearn import metrics

# loading the graph
directed_graph = nx.read_gml("polblogs.gml")
graph = directed_graph.to_undirected()
print("Number of nodes: %s" % nx.number_of_nodes(graph))
print("Number of edges: %s" % nx.number_of_edges(graph))

# get all the non existing edges before deleting random existing ones
non_existing_edges = list(nx.non_edges(graph))

# delete random existing edges
deleted_edges = helpers.delete_edges_randomly(graph, 10)

# Create a y_score array
ones_array = np.ones((len(deleted_edges),), dtype=np.int)
zeros_array = np.zeros((len(non_existing_edges),), dtype=np.int)
y_true = np.concatenate((zeros_array, ones_array), axis=0)

# performing link prediction methods

# Adamic Adar
aa_y_scores = helpers.get_y_score(nx.adamic_adar_index(graph, non_existing_edges),
                                  nx.adamic_adar_index(graph, deleted_edges))
aa_fpr, aa_tpr, aa_auc = helpers.get_roc_info(y_true, aa_y_scores)
aa_precision, aa_recall, aa_thresholds = metrics.precision_recall_curve(y_true, aa_y_scores)
aa_average_precision = metrics.average_precision_score(y_true, aa_y_scores)

# Preferential Attachment
pa_y_scores = helpers.get_y_score(nx.preferential_attachment(graph, non_existing_edges),
                                  nx.preferential_attachment(graph, deleted_edges))
pa_fpr, pa_tpr, pa_auc = helpers.get_roc_info(y_true, pa_y_scores)
pa_precision, pa_recall, pa_thresholds = metrics.precision_recall_curve(y_true, pa_y_scores)
pa_average_precision = metrics.average_precision_score(y_true, pa_y_scores)

# Jaccard Coefficient
jc_y_scores = helpers.get_y_score(nx.jaccard_coefficient(graph, non_existing_edges),
                                  nx.jaccard_coefficient(graph, deleted_edges))
jc_fpr, jc_tpr, jc_auc = helpers.get_roc_info(y_true, jc_y_scores)
jc_precision, jc_recall, jc_thresholds = metrics.precision_recall_curve(y_true, jc_y_scores)
jc_average_precision = metrics.average_precision_score(y_true, jc_y_scores)

# Resource Allocation Index
ra_y_scores = helpers.get_y_score(nx.resource_allocation_index(graph, non_existing_edges),
                                  nx.resource_allocation_index(graph, deleted_edges))
ra_fpr, ra_tpr, ra_auc = helpers.get_roc_info(y_true, ra_y_scores)
ra_precision, ra_recall, ra_thresholds = metrics.precision_recall_curve(y_true, ra_y_scores)
ra_average_precision = metrics.average_precision_score(y_true, ra_y_scores)

# Plot ROC curves
plt.figure()
plt.plot(aa_fpr, aa_tpr, label='Adamic Adar (AUC = %0.2f)' % aa_auc)
plt.plot(pa_fpr, pa_tpr, label='Preferential Attachment (AUC = %0.2f)' % pa_auc)
plt.plot(jc_fpr, jc_tpr, label='Jaccard Coefficient(AUC = %0.2f)' % jc_auc)
plt.plot(ra_fpr, ra_tpr, label='Resource Allocation (AUC = %0.2f)' % ra_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curves
plt.clf()
plt.plot(aa_recall, aa_precision,
         label='Adamic Adar (AUC = {0:0.2f})'.format(aa_average_precision))
plt.plot(pa_recall, pa_precision,
         label='Preferential Attachment (AUC = {0:0.2f})'.format(pa_average_precision))
plt.plot(jc_recall, jc_precision,
         label='Jaccard Coefficient (AUC = {0:0.2f})'.format(jc_average_precision))
plt.plot(ra_recall, ra_precision,
         label='Resource Allocation Index (AUC = {0:0.2f})'.format(ra_average_precision))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision Recall Curve')
plt.legend(loc="upper right")
plt.show()