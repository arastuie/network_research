import pickle
import helpers
import link_prediction_helpers as lp_helpers
import linear_regression_link_prediction as lr
import numpy as np
import os
import networkx as nx
from joblib import Parallel, delayed

# print("Reading in 200 random Facebook ego networks...")
#
# with open('/shared/DataSets/FacebookViswanath2009/egocentric/random_200_ego_nets.pckl', 'rb') as f:
#     ego_centric_networks, ego_nodes = pickle.load(f)

# with open('/shared/DataSets/FacebookViswanath2009/egocentric/first_50_ego.pckl', 'rb') as f:
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



# # path = '/home/marastu2/ego_network_research/main/Results/fb-lp-results/lower-6/temp'
# # path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files/lower-6/temp-3'
# path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files/after-6/temp-3'
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
#
# def run_link_prediction(index):
#     percent_aa = {}
#     percent_dcaa = {}
#     percent_cn = {}
#     percent_dccn = {}
#
#     for k in top_k_values:
#         percent_aa[k] = []
#         percent_dcaa[k] = []
#         percent_cn[k] = []
#         percent_dccn[k] = []
#
#     for ego_net_file in os.listdir('/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}'.format(index)):
#         with open('/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}/{1}'.format(index, ego_net_file), 'rb') as f:
#             egonet_snapshots, ego_node = pickle.load(f)
#
#         # aa, dcaa, cn, dccn = lp_helpers.run_adamic_adar_on_ego_net_ranking(egonet_snapshots, ego_node, top_k_values,
#         #                                                                    range(5))
#
#         aa, dcaa, cn, dccn = lp_helpers.run_adamic_adar_on_ego_net_ranking(egonet_snapshots, ego_node, top_k_values,
#                                                                            range(5, len(egonet_snapshots) - 1))
#
#         for k in top_k_values:
#             if len(aa[k]) > 0:
#                 percent_aa[k].append(np.mean(aa[k]))
#                 percent_dcaa[k].append(np.mean(dcaa[k]))
#                 percent_cn[k].append(np.mean(cn[k]))
#                 percent_dccn[k].append(np.mean(dccn[k]))
#
#     if len(percent_aa[top_k_values[0]]) > 0:
#         with open('{0}/{1}-lp-result.pckl'.format(path, index), 'wb') as f:
#             pickle.dump([percent_aa, percent_dcaa, percent_cn, percent_dccn], f, protocol=-1)
#     else:
#         print("No analysis in index {0}".format(index))
#
#
# Parallel(n_jobs=6)(delayed(run_link_prediction)(i) for i in range(0, 28))
#
# print("Merging all files...")
#
# percent_aa = {}
# percent_dcaa = {}
# percent_cn = {}
# percent_dccn = {}
#
# for k in top_k_values:
#     percent_aa[k] = []
#     percent_dcaa[k] = []
#     percent_cn[k] = []
#     percent_dccn[k] = []
#
# for result_file in os.listdir(path):
#     with open('{0}/{1}'.format(path, result_file), 'rb') as f:
#         aa, dcaa, cn, dccn = pickle.load(f)
#
#     for k in top_k_values:
#         percent_aa[k] = percent_aa[k] + aa[k]
#         percent_dcaa[k] = percent_dcaa[k] + dcaa[k]
#         percent_cn[k] = percent_cn[k] + cn[k]
#         percent_dccn[k] = percent_dccn[k] + dccn[k]
#
# with open('{0}/total-result.pckl'.format(path), 'wb') as f:
#     pickle.dump([percent_aa, percent_dcaa, percent_cn, percent_dccn], f, protocol=-1)
#
# for k in top_k_values:
#     print("For top {0}:".format(k))
#     aa = np.mean(percent_aa[k])
#     dcaa = np.mean(percent_dcaa[k])
#     aa_diff = dcaa - aa
#     aa_p_improve = aa_diff / aa
#     print("\taa -> {0}".format(aa))
#     print("\tdcaa -> {0}\n".format(dcaa))
#     print("\tdcaa - aa -> {0}".format(aa_diff))
#     print("\tdcaa percent improvement -> {0}".format(aa_p_improve))
#
#     cn = np.mean(percent_cn[k])
#     dccn = np.mean(percent_dccn[k])
#     cn_diff = dccn - cn
#     cn_p_improve = cn_diff / cn
#     print("\tcn -> {0}".format(cn))
#     print("\tdccn -> {0}\n".format(dccn))
#     print("\tdccn - cn -> {0}".format(dccn - cn))
#     print("\tdccn percent improvement -> {0}".format(cn_p_improve))










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
#     for ego_net_file in os.listdir('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes'):
#         percent_score_results = lp_helpers.run_link_prediction_comparison_on_directed_graph('/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/%s' % ego_net_file, triangle_type, top_k_values)
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
# for path in os.listdir('/shared/Results/EgocentricLinkPrediction/fb-lp-feature-extraction/fb_lp_features_reciprocal'):
#     aa_only_scores.append(list(lr.run_linear_regression('/shared/Results/EgocentricLinkPrediction/fb-lp-feature-extraction/fb_lp_features_reciprocal/{0}'.format(path), True)))
#     both_scores.append(list(lr.run_linear_regression('/shared/Results/EgocentricLinkPrediction/fb-lp-feature-extraction/fb_lp_features_reciprocal/{0}'.format(path), False)))
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









############## With CCLP & CAR ##################
# # path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files/lower-6/temp-4'
# path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files/after-6/temp-4'
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
#
# def run_link_prediction(index):
#     percent_cclp = {}
#     # percent_dcaa = {}
#     percent_car = {}
#     # percent_dccn = {}
#
#     for k in top_k_values:
#         percent_cclp[k] = []
#         # percent_dcaa[k] = []
#         percent_car[k] = []
#         # percent_dccn[k] = []
#
#     for ego_net_file in os.listdir('/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}'.format(index)):
#         with open('/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}/{1}'.format(index, ego_net_file), 'rb') as f:
#             egonet_snapshots, ego_node = pickle.load(f)
#
#         # cclp, dcaa, car, dccn = lp_helpers.run_adamic_adar_on_ego_net_ranking_with_cclp_and_car(egonet_snapshots,
#         #                                                                                         ego_node, top_k_values,
#         #                                                                                         range(5))
#
#         # cclp, dcaa, car, dccn = lp_helpers.run_adamic_adar_on_ego_net_ranking_with_cclp_and_car(egonet_snapshots,
#         #                                                                                         ego_node, top_k_values,
#         #                                                                                         range(5, len(egonet_snapshots) - 1))
#
#         cclp, car = lp_helpers.run_adamic_adar_on_ego_net_ranking_only_cclp_and_car(egonet_snapshots, ego_node, top_k_values,
#                                                                                   range(5, len(egonet_snapshots) - 1))
#
#         for k in top_k_values:
#             if len(cclp[k]) > 0:
#                 percent_cclp[k].append(np.mean(cclp[k]))
#                 # percent_dcaa[k].append(np.mean(dcaa[k]))
#                 percent_car[k].append(np.mean(car[k]))
#                 # percent_dccn[k].append(np.mean(dccn[k]))
#
#     if len(percent_cclp[top_k_values[0]]) > 0:
#         # with open('{0}/{1}-lp-result.pckl'.format(path, index), 'wb') as f:
#         #     pickle.dump([percent_cclp, percent_dcaa, percent_car, percent_dccn], f, protocol=-1)
#
#         with open('{0}/{1}-lp-result.pckl'.format(path, index), 'wb') as f:
#             pickle.dump([percent_cclp, percent_car], f, protocol=-1)
#
#         print("Done with index {0}".format(index))
#     else:
#         print("No analysis in index {0}".format(index))
#
#
# Parallel(n_jobs=28)(delayed(run_link_prediction)(i) for i in range(0, 28))
#
# print("Merging all files...")
#
# percent_cclp = {}
# # percent_dcaa = {}
# percent_car = {}
# # percent_dccn = {}
#
# for k in top_k_values:
#     percent_cclp[k] = []
#     # percent_dcaa[k] = []
#     percent_car[k] = []
#     # percent_dccn[k] = []
#
# for result_file in os.listdir(path):
#     # with open('{0}/{1}'.format(path, result_file), 'rb') as f:
#     #     cclp, dcaa, car, dccn = pickle.load(f)
#     with open('{0}/{1}'.format(path, result_file), 'rb') as f:
#         cclp, car = pickle.load(f)
#
#     for k in top_k_values:
#         percent_cclp[k] = percent_cclp[k] + cclp[k]
#         # percent_dcaa[k] = percent_dcaa[k] + dcaa[k]
#         percent_car[k] = percent_car[k] + car[k]
#         # percent_dccn[k] = percent_dccn[k] + dccn[k]
#
# # with open('{0}/total-result.pckl'.format(path), 'wb') as f:
# #     pickle.dump([percent_cclp, percent_dcaa, percent_car, percent_dccn], f, protocol=-1)
#
# with open('{0}/total-result.pckl'.format(path), 'wb') as f:
#     pickle.dump([percent_cclp, percent_car], f, protocol=-1)
#
# # for k in top_k_values:
# #     print("For top {0}:".format(k))
# #     cclp = np.mean(percent_cclp[k])
# #     dcaa = np.mean(percent_dcaa[k])
# #     cclp_diff = dcaa - cclp
# #     cclp_p_improve = cclp_diff / cclp
# #     print("\tcclp -> {0}".format(cclp))
# #     print("\tdcaa -> {0}\n".format(dcaa))
# #     print("\tdcaa - cclp -> {0}".format(cclp_diff))
# #     print("\tdcaa percent improvement -> {0}".format(cclp_p_improve))
# #
# #     dccn = np.mean(percent_dccn[k])
# #     cclp_diff = dccn - cclp
# #     cclp_p_improve = cclp_diff / cclp
# #     print("\tcclp -> {0}".format(cclp))
# #     print("\tdccn -> {0}\n".format(dccn))
# #     print("\tdccn - cclp -> {0}".format(dccn - cclp))
# #     print("\tdccn percent improvement -> {0}".format(cclp_p_improve))
# #
# #     print("For top {0}:".format(k))
# #     car = np.mean(percent_car[k])
# #     dcaa = np.mean(percent_dcaa[k])
# #     car_diff = dcaa - car
# #     car_p_improve = car_diff / car
# #     print("\tcar -> {0}".format(car))
# #     print("\tdcaa -> {0}\n".format(dcaa))
# #     print("\tdcaa - car -> {0}".format(car_diff))
# #     print("\tdcaa percent improvement -> {0}".format(car_p_improve))
# #
# #     dccn = np.mean(percent_dccn[k])
# #     car_diff = dccn - car
# #     car_p_improve = car_diff / car
# #     print("\tcar -> {0}".format(car))
# #     print("\tdccn -> {0}\n".format(dccn))
# #     print("\tdccn - car -> {0}".format(dccn - car))
# #     print("\tdccn percent improvement -> {0}".format(car_p_improve))






#######################  Test new lp methods on FB #########################
datasets_directory = '/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets'
path_before_pymk = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files/lower-6/temp-5'
path_after_pymk = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files/after-6/temp-5'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]


def run_link_prediction_on_test_method(index, num_ego_per_index):
    percent_test_before = {}
    percent_test_after = {}

    for k in top_k_values:
        percent_test_before[k] = []
        percent_test_after[k] = []

    count = 0
    for ego_net_file in os.listdir('{0}/{1}'.format(datasets_directory, index)):
        with open('{0}/{1}/{2}'.format(datasets_directory, index, ego_net_file), 'rb') as f:
            egonet_snapshots, ego_node = pickle.load(f)

        test_score_before = lp_helpers.run_link_prediction_on_test_method(egonet_snapshots, ego_node, top_k_values,
                                                                          range(5))

        test_score_after = lp_helpers.run_link_prediction_on_test_method(egonet_snapshots, ego_node, top_k_values,
                                                                         range(5, len(egonet_snapshots) - 1))

        for k in top_k_values:
            if len(test_score_before[k]) > 0:
                percent_test_before[k].append(np.mean(test_score_before[k]))
                count += 1

            if len(test_score_after[k]) > 0:
                percent_test_after[k].append(np.mean(test_score_after[k]))

        if count >= num_ego_per_index:
            break

    if len(percent_test_before[top_k_values[0]]) > 0:
        with open('{0}/{1}-lp-result.pckl'.format(path_before_pymk, index), 'wb') as f:
            pickle.dump(percent_test_before, f, protocol=-1)

    if len(percent_test_after[top_k_values[0]]) > 0:
        with open('{0}/{1}-lp-result.pckl'.format(path_after_pymk, index), 'wb') as f:
            pickle.dump(percent_test_after, f, protocol=-1)

        print("Done with index {0}".format(index))
    else:
        print("No analysis in index {0}".format(index))


Parallel(n_jobs=28)(delayed(run_link_prediction_on_test_method)(i, 50) for i in range(0, 28))

print("Merging all files...")

for path in [path_before_pymk, path_after_pymk]:
    percent_test = {}

    for k in top_k_values:
        percent_test[k] = []

    for result_file in os.listdir(path):
        if result_file == 'total-result.pckl':
            continue

        with open('{0}/{1}'.format(path, result_file), 'rb') as f:
            test_score = pickle.load(f)

        for k in top_k_values:
            percent_test[k] = percent_test[k] + test_score[k]

    with open('{0}/total-result.pckl'.format(path), 'wb') as f:
        pickle.dump(percent_test, f, protocol=-1)

# reading result for final evaluation
cnt = 0
for path in [path_before_pymk, path_after_pymk]:

    with open('{0}/total-result.pckl'.format(path), 'rb') as f:
        percent_test = pickle.load(f)

    if cnt == 0:
        print("\nResults for before PYMK:")
    else:
        print("\nResults for after PYMK:")

    for k in top_k_values:
        print("\tTop {0} K -> {1}".format(k, np.mean(percent_test[k])))

    cnt += 1
