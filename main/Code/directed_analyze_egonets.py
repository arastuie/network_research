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
import gplus_helpers as gplus
# import gplus_hop_degree_directed_analysis_cdf as analyzer


# ************************************************************************* #
# ******************** Local Degree Empirical Analysis ******************** #
# ************************************************************************* #
def run_empirical_local_degree_parallel_analysis(egonet_files_path, results_base_path, num_process):
    all_egonets = set(os.listdir(egonet_files_path))
    analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets')).union(os.listdir(results_base_path +
                                                                                                'skipped_egonets'))
    egonets_to_analyze = list(all_egonets - analyzed_egonets)
    np.random.shuffle(egonets_to_analyze)

    print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(dgh.run_local_degree_empirical_analysis)
                                 (ego_net_file, results_base_path, egonet_files_path) for ego_net_file in all_egonets)


# **** Google+ Local Degree Empirical Test **** #
run_empirical_local_degree_parallel_analysis(gplus.egonet_files_path, gplus.local_degree_empirical_results_path, 6)


# **** Flickr Local Degree Empirical Test **** #
# run_empirical_local_degree_parallel_analysis(flickr.egonet_files_path, flickr.local_degree_empirical_results_path, 20)


# # Digg Directed Empirical Test
# all_egonets = set(os.listdir(digg.digg_egonets_file_path))
# analyzed_egonets = set(os.listdir(digg.digg_results_file_path + 'analyzed_egonets')).union(
#     os.listdir(digg.digg_results_file_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# Parallel(n_jobs=17)(delayed(digg.run_local_degree_empirical_analysis)(ego_net_file) for ego_net_file in all_egonets)


# Flickr Link prediction
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
# all_egonets = set(os.listdir(flickr.flickr_growth_egonets_path))
# analyzed_egonets = set(os.listdir(flickr.flickr_growth_lp_result_path + 'analyzed_egonets')).union(
#     os.listdir(flickr.flickr_growth_lp_result_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=24)(delayed(dh.run_directed_link_prediction)
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


# # Google+ Link prediction
# gplus_egonet_files_path = "/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/egonets-w-snapshots"
# gplus_egonet_results_path = "/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files-1/"
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
# all_egonets = set(os.listdir(gplus_egonet_files_path))
# analyzed_egonets = set(os.listdir(gplus_egonet_results_path + 'analyzed_egonets')).union(
#     os.listdir(gplus_egonet_results_path + 'skipped_egonets'))
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=5)(delayed(dh.run_directed_link_prediction)
#                     (ego_net_file, top_k_values, gplus_egonet_files_path, gplus_egonet_results_path)
#                     for ego_net_file in egonets_to_analyze)


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

Parallel(n_jobs=6)(delayed(dgh.empirical_triad_links_formed_ratio)
                    (ego_net_file, egonets_path, result_path)for ego_net_file in egonets_to_analyze)
