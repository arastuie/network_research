import os
import pickle
import numpy as np
# import helpers as h
import flickr_helpers as flickr
import digg_net_helpers as digg
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import link_prediction_helpers as dh
import directed_graphs_helpers as dgh
# import gplus_hop_degree_directed_analysis_cdf as analyzer

# Google+ Empirical Test
# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/test-2/'
# data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes'
# triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
# temp_skipped_files = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/temp'
# all_egonets = os.listdir(data_file_base_path)
# np.random.shuffle(all_egonets)
# Parallel(n_jobs=6)(delayed(analyzer.gplus_run_hop_degree_directed_analysis)(ego_net_file)
#                    for ego_net_file in all_egonets)


# Flickr Empirical Test
# all_egonets = set(os.listdir(flickr.flickr_growth_egonets_path))
# analyzed_egonets = set(os.listdir(flickr.flickr_growth_empirical_result_path + 'analyzed_egonets')).union(
#     os.listdir(flickr.flickr_growth_empirical_result_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# Parallel(n_jobs=20)(delayed(flickr.run_local_degree_empirical_analysis)(ego_net_file) for ego_net_file in all_egonets)


# # Digg Directed Empirical Test
# all_egonets = set(os.listdir(digg.digg_egonets_file_path))
# analyzed_egonets = set(os.listdir(digg.digg_results_file_path + 'analyzed_egonets')).union(
#     os.listdir(digg.digg_results_file_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# Parallel(n_jobs=17)(delayed(digg.run_local_degree_empirical_analysis)(ego_net_file) for ego_net_file in all_egonets)


# Parallel(n_jobs=15)(delayed(dh.run_link_prediction_comparison_on_directed_graph_all_types)(ego_net_file, top_k_values)
#                     for ego_net_file in os.listdir(data_file_base_path))

# Parallel(n_jobs=16)(delayed(dh.run_link_prediction_comparison_on_directed_graph_all_types_based_on_empirical)
#                     (ego_net_file, top_k_values) for ego_net_file in os.listdir(data_file_base_path))

# Parallel(n_jobs=10)(delayed(dh.run_link_prediction_comparison_on_directed_graph_combined_types)
#                    (ego_net_file, top_k_values) for ego_net_file in os.listdir(temp_skipped_files))

# all_files = set(os.listdir(data_file_base_path))
# analyzed_files = set(os.listdir(result_file_base_path + 'analyzed_egonets'))
# skipped_files = set(os.listdir(result_file_base_path + 'skipped_egonets'))
#
# left_files = all_files - analyzed_files
# left_files = left_files - skipped_files
#
# print(len(left_files))
#
# Parallel(n_jobs=24)(delayed(dh.run_link_prediction_comparison_on_directed_graph_combined_types)
#                     (ego_net_file, top_k_values) for ego_net_file in os.listdir(result_file_base_path + 'skipped_egonets'))


# res_from_prev_analysis = "/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/test-2/results"
# result_file_base_3 = "/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/test-3-car-cclp/"
# all_files = set(os.listdir(res_from_prev_analysis))
# analyzed_files = set(os.listdir(result_file_base_3 + 'analyzed_egonets'))
# skipped_files = set(os.listdir(result_file_base_3 + 'skipped_egonets'))
#
# left_files = all_files - analyzed_files
# left_files = list(left_files - skipped_files)
# np.random.shuffle(left_files)
#
# Parallel(n_jobs=6)(delayed(dh.run_link_prediction_comparison_on_directed_graph_combined_types_only_car_and_cclp)
#                    (ego_net_file, top_k_values) for ego_net_file in left_files)

## Flickr Link prediction
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
# all_egonets = set(os.listdir(flickr.flickr_growth_egonets_path))
# analyzed_egonets = set(os.listdir(flickr.flickr_growth_lp_result_path + 'analyzed_egonets')).union(
#     os.listdir(flickr.flickr_growth_lp_result_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=12)(delayed(dh.run_directed_link_prediction)
#                     (ego_net_file, top_k_values, flickr.flickr_growth_egonets_path, flickr.flickr_growth_lp_result_path)
#                     for ego_net_file in egonets_to_analyze)



# ### Digg Link prediction
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
# all_egonets = set(os.listdir(digg.digg_egonets_file_path))
# analyzed_egonets = set(os.listdir(digg.digg_results_lp_file_path + 'analyzed_egonets')).union(
#     os.listdir(digg.digg_results_lp_file_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=24)(delayed(dh.run_directed_link_prediction)
#                     (ego_net_file, top_k_values, digg.digg_egonets_file_path, digg.digg_results_lp_file_path)
#                     for ego_net_file in egonets_to_analyze)


# ### Google+ Empirical Triad Links Formed Ratio Test
# data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/'
# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/triad-link-formed-ratio/' \
#                         'pickle-files/'
#
# all_egonets = set(os.listdir(data_file_base_path))
# analyzed_egonets = set(os.listdir(result_file_base_path + 'analyzed_egonets')).union(os.listdir(result_file_base_path +
#                                                                                                 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=12)(delayed(dgh.empirical_triad_links_formed_ratio)
#                    (ego_net_file, data_file_base_path, result_file_base_path)for ego_net_file in egonets_to_analyze)


# ### Flickr Empirical Triad Links Formed Ratio Test
# egonets_path = flickr.flickr_growth_egonets_path + '/'
# all_egonets = set(os.listdir(egonets_path))
# analyzed_egonets = set(os.listdir(flickr.flickr_growth_empirical_triad_ratio_result_path + 'analyzed_egonets')).union(
#     os.listdir(flickr.flickr_growth_empirical_triad_ratio_result_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=12)(delayed(dgh.empirical_triad_links_formed_ratio)
#                     (ego_net_file, egonets_path, flickr.flickr_growth_empirical_triad_ratio_result_path)
#                     for ego_net_file in egonets_to_analyze)



# ### Digg Empirical Triad Links Formed Ratio Test
# egonets_path = digg.digg_egonets_file_path + '/'
# result_path = digg.digg_empirical_triad_ratio_result_path
#
# all_egonets = set(os.listdir(egonets_path))
# analyzed_egonets = set(os.listdir(result_path + 'analyzed_egonets')).union(os.listdir(result_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=12)(delayed(dgh.empirical_triad_links_formed_ratio)
#                     (ego_net_file, egonets_path, result_path)for ego_net_file in egonets_to_analyze)


# ### Google+ Empirical Triad Links Formed Ratio Test
egonets_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/egonets-w-snapshots/'
result_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/triad-link-formed-ratio/pickle-files/'

all_egonets = set(os.listdir(egonets_path))
analyzed_egonets = set(os.listdir(result_path + 'analyzed_egonets')).union(os.listdir(result_path + 'skipped_egonets'))
egonets_to_analyze = list(all_egonets - analyzed_egonets)
np.random.shuffle(egonets_to_analyze)

print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

Parallel(n_jobs=12)(delayed(dgh.empirical_triad_links_formed_ratio)
                    (ego_net_file, egonets_path, result_path)for ego_net_file in egonets_to_analyze)