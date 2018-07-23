import os
import numpy as np
import gplus_helpers as gplus
import flickr_helpers as flickr
import digg_helpers as digg
from joblib import Parallel, delayed
import directed_graphs_helpers as dgh
import link_prediction_evaluator as lpe


# ************************************************************************* #
# ******************** Local Degree Empirical Analysis ******************** #
# ************************************************************************* #
def run_parallel_local_degree_empirical_analysis(egonet_files_path, results_base_path, num_process, skip_over_100k):
    all_egonets = set(os.listdir(egonet_files_path))
    if skip_over_100k:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets')).union(os.listdir(results_base_path +
                                                                                                    'skipped_egonets'))
    else:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)
    np.random.shuffle(egonets_to_analyze)

    print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(dgh.run_local_degree_empirical_analysis)
                                 (ego_net_file, results_base_path, egonet_files_path, skip_over_100k)
                                 for ego_net_file in all_egonets)


# **** Google+ **** #
# run_parallel_local_degree_empirical_analysis(gplus.egonet_files_path, gplus.local_degree_empirical_results_path, 6,
#                                              skip_over_100k=False)
#
# dgh.plot_local_degree_empirical_results(gplus.local_degree_empirical_results_path,
#                                         gplus.local_degree_empirical_plot_path, gather_individual_results=False)
#
# dgh.plot_local_degree_empirical_cdf(gplus.local_degree_empirical_results_path, gplus.local_degree_empirical_plot_path,
#                                     triangle_types='all', separete_in_out_degree=False,
#                                     gather_individual_results=True)


# **** Flickr **** #
# run_parallel_local_degree_empirical_analysis(flickr.egonet_files_path, flickr.local_degree_empirical_results_path, 6,
#                                              skip_over_100k=False)
#
# dgh.plot_local_degree_empirical_results(flickr.local_degree_empirical_results_path,
#                                         flickr.local_degree_empirical_plot_path, gather_individual_results=True)
#
# dgh.plot_local_degree_empirical_cdf(flickr.local_degree_empirical_results_path,
#                                     flickr.local_degree_empirical_plot_path, triangle_types='all',
#                                     separete_in_out_degree=False, gather_individual_results=True)


# **** Digg **** #
# run_parallel_local_degree_empirical_analysis(digg.egonet_files_path,
#                                              digg.directed_local_degree_empirical_results_path, 17,
#                                              skip_over_100k=False)
#
# dgh.plot_local_degree_empirical_results(digg.directed_local_degree_empirical_results_path,
#                                         digg.directed_local_degree_empirical_plot_path,
#                                         gather_individual_results=True)
#
# dgh.plot_local_degree_empirical_cdf(digg.directed_local_degree_empirical_results_path,
#                                     digg.directed_local_degree_empirical_plot_path, triangle_types='all',
#                                     separete_in_out_degree=False, gather_individual_results=True)


# ************************************************************************* #
# *************** Triad Links Formed Ratio Empirical Analysis ************* #
# ************************************************************************* #
def run_parallel_triad_ratio_analysis(egonet_files_path, results_base_path, num_process, skip_over_100k):
    all_egonets = set(os.listdir(egonet_files_path))
    if skip_over_100k:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets')).union(os.listdir(results_base_path +
                                                                                                    'skipped_egonets'))
    else:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)
    np.random.shuffle(egonets_to_analyze)

    print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(dgh.empirical_triad_links_formed_ratio)
                                 (ego_net_file, egonet_files_path, results_base_path, skip_over_100k)
                                 for ego_net_file in egonets_to_analyze)


# **** Google+ **** #
# run_parallel_triad_ratio_analysis(gplus.egonet_files_path, gplus.triad_ratio_empirical_results_path, 4,
#                                   skip_over_100k=False)
#
# dgh.empirical_triad_list_formed_ratio_results_plot(gplus.triad_ratio_empirical_results_path,
#                                                    gplus.triad_ratio_empirical_plots_path,
#                                                    gather_individual_results=True)
#
# **** Flickr **** #
# run_parallel_triad_ratio_analysis(flickr.egonet_files_path, flickr.triad_ratio_empirical_results_path, 4,
#                                   skip_over_100k=False)
#
# dgh.empirical_triad_list_formed_ratio_results_plot(flickr.triad_ratio_empirical_results_path,
#                                                    flickr.triad_ratio_empirical_plots_path,
#                                                    gather_individual_results=True)
#
# **** Digg **** #
# run_parallel_triad_ratio_analysis(digg.egonet_files_path, digg.triad_ratio_empirical_results_path, 12,
#                                   skip_over_100k=False)
#
# dgh.empirical_triad_list_formed_ratio_results_plot(digg.triad_ratio_empirical_results_path,
#                                                    digg.triad_ratio_empirical_plots_path,
#                                                    gather_individual_results=True)


# ************************************************************************* #
# ************************ Link Prediction Analysis *********************** #
# ************************************************************************* #
def run_parallel_link_prediction_analysis(egonet_files_path, results_base_path, num_process, skip_over_100k):
    top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
    all_egonets = set(os.listdir(egonet_files_path))
    if skip_over_100k:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets')).union(os.listdir(results_base_path +
                                                                                                    'skipped_egonets'))
    else:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)
    np.random.shuffle(egonets_to_analyze)

    print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(dgh.run_directed_link_prediction)
                                 (ego_net_file, top_k_values, egonet_files_path, results_base_path)
                                 for ego_net_file in egonets_to_analyze)


# The methods to be compared for improvements plots
comparison_pairs = [('cn', 'dccn'), ('aa', 'dcaa')]

# **** Google+ **** #
run_parallel_link_prediction_analysis(gplus.egonet_files_path, gplus.lp_results_path, 6, skip_over_100k=False)
# lpe.calculate_lp_performance(gplus.lp_results_path, gather_individual_results=True)
# lpe.plot_percent_improvements(gplus.lp_results_path, gplus.lp_plots_path, comparison_pairs,
#                               gather_individual_results=True)

# **** Flickr **** #
run_parallel_link_prediction_analysis(flickr.egonet_files_path, flickr.lp_results_path, 6, skip_over_100k=False)
# lpe.calculate_lp_performance(flickr.lp_results_path, gather_individual_results=True)
# lpe.plot_percent_improvements(flickr.lp_results_path, flickr.lp_plots_path, comparison_pairs,
#                               gather_individual_results=True)

# **** Digg **** #
# run_parallel_link_prediction_analysis(digg.egonet_files_path, digg.lp_results_file_path, 6, skip_over_100k=False)
# lpe.calculate_lp_performance(digg.lp_results_file_path, gather_individual_results=False)
# lpe.plot_percent_improvements(digg.lp_results_file_path, digg.lp_plots_path, comparison_pairs,
#                               gather_individual_results=True)
