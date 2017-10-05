import pickle
import helpers
import link_prediction_helpers as lp_helpers
import linear_regression_link_prediction as lr
import numpy as np
import os
import networkx as nx
from joblib import Parallel, delayed

print("Reading in 200 random Facebook ego networks...")

with open('../Data/random_200_ego_nets.pckl', 'rb') as f:
    ego_centric_networks, ego_nodes = pickle.load(f)

# with open('../Data/first_50_ego.pckl', 'rb') as f:
#     ego_centric_networks, ego_nodes = pickle.load(f)

# print("Networks in!")

# lp_results = []
# # for w in np.arange(10, 20, 2):
# for i in range(len(ego_centric_networks)):
#     scores = lp_helpers.run_adamic_adar_on_ego_net(ego_centric_networks[i], ego_nodes[i])
#
#     if scores is not None:
#         lp_results.append(scores)
#
# lp_helpers.plot_auroc_hist(lp_results)
# lp_helpers.plot_pr_hist(lp_results)

# fb_graph = helpers.read_facebook_graph()
#
# orig_snapshots = []
#
# oldest_timestamp = 1157454929
# seconds_in_90_days = 7776000
# for i in range(10):
#     orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in fb_graph.edges(data=True)
#                                     if d['timestamp'] < (oldest_timestamp + seconds_in_90_days * i)]))
# orig_snapshots.append(fb_graph)
# fb_graph = None

percent_aa = {}
percent_dcaa = {}

top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
for k in top_k_values:
    percent_aa[k] = []
    percent_dcaa[k] = []

for i in range(len(ego_centric_networks)):
    aa, dcaa = lp_helpers.run_adamic_adar_on_ego_net_ranking(ego_centric_networks[i], ego_nodes[i], top_k_values)

    for k in top_k_values:
        for m in aa[k]:
            percent_aa[k].append(m)

    for k in top_k_values:
        for n in dcaa[k]:
            percent_dcaa[k].append(n)

    print('{0}'.format(i), end='\r')

for k in top_k_values:
    print("For top {0}:".format(k))
    aa = np.mean(percent_aa[k])
    print("\taa -> {0}".format(aa))
    dcaa = np.mean(percent_dcaa[k])
    print("\tdcaa -> {0}\n".format(dcaa))
    print("\tdcaa - aa -> {0}".format(dcaa - aa))


# def run_directed_link_prediction_analysis(triangle_type):
#     percent_scores = {
#         'cn': {},
#         'dccn_i': {},
#         'dccn_o': {},
#         'aa_i': {},
#         'aa_o': {},
#         'dcaa_i': {},
#         'dcaa_o': {}
#     }
#
#     top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#     for k in top_k_values:
#         for ps in percent_scores.keys():
#             percent_scores[ps][k] = []
#     i = 0
#     for ego_net_file in os.listdir('../Data/gplus-ego/first-hop-nodes'):
#         percent_score_results = lp_helpers.run_link_prediction_comparison_on_directed_graph('../Data/gplus-ego/first-hop-nodes/%s' % ego_net_file, triangle_type, top_k_values)
#
#         if percent_score_results is None:
#             continue
#
#         any_result = False
#
#         for ps in percent_scores.keys():
#             for k in top_k_values:
#                 for m in percent_score_results[ps][k]:
#                     any_result = True
#                     percent_scores[ps][k].append(m)
#
#         if any_result:
#             i = i + 1
#             print('{0} -> {1}'.format(triangle_type, i))
#
#     with open("../Results/directed_lp_result/{0}-directed_lp_result.txt".format(triangle_type), "w") as f:
#         for k in top_k_values:
#             f.write("\nFor top {0}:\n".format(k))
#             for ps in percent_scores:
#                 f.write("\t{0} -> {1}\n".format(ps, np.mean(percent_scores[ps][k])))


# triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
# Parallel(n_jobs=9)(delayed(run_directed_link_prediction_analysis)(triangle_type) for triangle_type in triangle_types)

# run_directed_link_prediction_analysis('T09')

# degree_formation = None
# for i in range(len(ego_centric_networks)):
#     degrees = lp_helpers.ego_net_link_formation_hop_degree(ego_centric_networks[i], ego_nodes[i])
#
#     if degrees is not None:
#         if degree_formation is None:
#             degree_formation = np.array(degrees)
#         else:
#             degree_formation = np.concatenate((degree_formation, np.array(degrees)))
#
#     print('{0}'.format(i), end='\r')
#
# lp_helpers.plot_degree_scatter(degree_formation)

# index_comparison = None
# for i in range(len(ego_centric_networks)):
#     comparison = lp_helpers.run_adamic_adar_on_ego_net_ranking_plot(ego_centric_networks[i], ego_nodes[i])
#
#     if comparison is not None:
#         if index_comparison is None:
#             index_comparison = np.array(comparison)
#         else:
#             index_comparison = np.concatenate((index_comparison, np.array(comparison)))
#
#     print('{0}'.format(i), end='\r')
#
# lp_helpers.plot_degree_scatter(index_comparison)

# aa_only_scores = []
# both_scores = []
# cnt = 1
# for path in os.listdir('../Data/fb_lp_features_reciprocal'):
#     aa_only_scores.append(list(lr.run_linear_regression('../Data/fb_lp_features_reciprocal/{0}'.format(path), True)))
#     both_scores.append(list(lr.run_linear_regression('../Data/fb_lp_features_reciprocal/{0}'.format(path), False)))
#     print(cnt, end='\r')
#     cnt += 1
#
# aa_only_scores = np.array(aa_only_scores)
# print("AA Only:")
# print("Mean AA Coefficient: {0}".format(np.mean(aa_only_scores[:, 0])))
# print("Mean AUROC: {0}".format(np.mean(aa_only_scores[:, 1])))
# print("Mean Avg Precision: {0}".format(np.mean(aa_only_scores[:, 2])))
#
# both_scores = np.array(both_scores)
# print("\nAA With DCAA:")
# print("Mean AA Coefficient: {0}".format(np.mean(both_scores[:, 0])))
# print("Mean DCAA Coefficient: {0}".format(np.mean(both_scores[:, 1])))
# print("Mean AUROC: {0}".format(np.mean(both_scores[:, 2])))
# print("Mean Avg Precision: {0}".format(np.mean(both_scores[:, 3])))

# aa_only_scores = np.array(aa_only_scores)
# print("AA Only:")
# print("Mean AA Coefficient: {0}".format(np.mean(aa_only_scores[:, 0])))
# print("Mean AA Coefficient ** 2: {0}".format(np.mean(aa_only_scores[:, 1])))
# print("Mean AUROC: {0}".format(np.mean(aa_only_scores[:, 2])))
# print("Mean Avg Precision: {0}".format(np.mean(aa_only_scores[:, 3])))
#
# both_scores = np.array(both_scores)
# print("\nAA With DCAA:")
# print("Mean AA Coefficient: {0}".format(np.mean(both_scores[:, 0])))
# print("Mean AA Coefficient ** 2: {0}".format(np.mean(both_scores[:, 2])))
# print("Mean DCAA Coefficient: {0}".format(np.mean(both_scores[:, 1])))
# print("Mean DCAA Coefficient ** 2: {0}".format(np.mean(both_scores[:, 3])))
# print("Mean AA * DCAA Coefficient: {0}".format(np.mean(both_scores[:, 4])))
# print("Mean AUROC: {0}".format(np.mean(both_scores[:, 5])))
# print("Mean Avg Precision: {0}".format(np.mean(both_scores[:, 6])))
