import os
import shutil
import numpy as np
import gplus_helpers as gplus
import flickr_helpers as flickr
import digg_helpers as digg
from joblib import Parallel, delayed
import directed_graphs_helpers as dgh
import link_prediction_evaluator as lpe
import directed_ml_helpers as dmlh


# ************************************************************************* #
# ******************** Local Degree Empirical Analysis ******************** #
# ************************************************************************* #
def run_parallel_local_degree_empirical_analysis(egonet_files_path, results_base_path, num_process, skip_over_100k,
                                                 log_degree=False, skip_snaps=True, normalize=True):
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)
        os.makedirs(results_base_path + 'all-scores')
        os.makedirs(results_base_path + 'analyzed_egonets')
        os.makedirs(results_base_path + 'skipped_egonets')
        os.makedirs(results_base_path + 'temp-analyses-start')
        for i in range(1, 10):
            os.makedirs(results_base_path + '/T0{}'.format(i))

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
                                 (ego_net_file, results_base_path, egonet_files_path, log_degree, skip_snaps, normalize,
                                  skip_over_100k=skip_over_100k) for ego_net_file in all_egonets)


# **** Google+ **** #
# run_parallel_local_degree_empirical_analysis(gplus.egonet_files_path,
#                                              gplus.local_degree_empirical_results_base_path + 'pickle-files-1/', 2,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=True, normalize=True)

# run_parallel_local_degree_empirical_analysis(gplus.egonet_files_path,
#                                              gplus.local_degree_empirical_results_base_path + 'pickle-files-2/', 2,
#                                              skip_over_100k=True, log_degree=True, skip_snaps=False, normalize=False)

# run_parallel_local_degree_empirical_analysis(gplus.egonet_files_path,
#                                              gplus.local_degree_empirical_results_base_path + 'pickle-files-3/', 2,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=False, normalize=True)

# run_parallel_local_degree_empirical_analysis(gplus.egonet_files_path,
#                                              gplus.local_degree_empirical_results_base_path + 'pickle-files-4/', 12,
#                                              skip_over_100k=True, log_degree=True, skip_snaps=True, normalize=False)

# run_parallel_local_degree_empirical_analysis(gplus.egonet_files_path,
#                                              gplus.local_degree_empirical_results_base_path + 'pickle-files-5/', 10,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=True, normalize=False)


# dgh.plot_local_degree_empirical_results(gplus.local_degree_empirical_results_base_path + 'pickle-files-4/',
#                                         gplus.local_degree_empirical_base_plot_path + 'plots-4/',
#                                         gather_individual_results=False)

# dgh.plot_local_degree_empirical_cdf(gplus.local_degree_empirical_results_base_path + 'pickle-files-1/',
#                                     gplus.local_degree_empirical_base_plot_path + 'plots-1/',
#                                     triangle_types=['T01'], separete_in_out_degree=True,
#                                     gather_individual_results=True)

# dgh.local_degree_empirical_result_comparison(gplus.local_degree_empirical_results_base_path + 'pickle-files-1/',
#                                              include_conf_intervals=True,
#                                              gather_individual_results=False)


# **** Flickr **** #
# run_parallel_local_degree_empirical_analysis(flickr.egonet_files_path,
#                                              flickr.local_degree_empirical_results_base_path + 'test1/', 2,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=True, normalize=True)

# run_parallel_local_degree_empirical_analysis(flickr.egonet_files_path,
#                                              flickr.local_degree_empirical_results_base_path + 'test2/', 2,
#                                              skip_over_100k=True, log_degree=True, skip_snaps=False, normalize=False)

# run_parallel_local_degree_empirical_analysis(flickr.egonet_files_path,
#                                              flickr.local_degree_empirical_results_base_path + 'test3/', 1,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=False, normalize=True)

# run_parallel_local_degree_empirical_analysis(flickr.egonet_files_path,
#                                              flickr.local_degree_empirical_results_base_path + 'test4/', 16,
#                                              skip_over_100k=True, log_degree=True, skip_snaps=True, normalize=False)

# run_parallel_local_degree_empirical_analysis(flickr.egonet_files_path,
#                                              flickr.local_degree_empirical_results_base_path + 'test5/', 12,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=True, normalize=False)



# dgh.plot_local_degree_empirical_results(flickr.local_degree_empirical_results_base_path + 'test4/',
#                                         flickr.local_degree_empirical_plot_base_path + 'test4/',
#                                         gather_individual_results=False)

# dgh.plot_local_degree_empirical_cdf(flickr.local_degree_empirical_results_base_path + 'test1/',
#                                     flickr.local_degree_empirical_plot_base_path + 'test1/', triangle_types=['T01'],
#                                     separete_in_out_degree=True, gather_individual_results=True)

# dgh.local_degree_empirical_result_comparison(flickr.local_degree_empirical_results_base_path + 'test1/',
#                                              include_conf_intervals=True, gather_individual_results=False)


# **** Digg **** #
# run_parallel_local_degree_empirical_analysis(digg.egonet_files_path,
#                                              digg.directed_local_degree_empirical_base_results_path + 'directed/', 2,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=True, normalize=True)

# run_parallel_local_degree_empirical_analysis(digg.egonet_files_path,
#                                              digg.directed_local_degree_empirical_base_results_path + 'directed-1/', 2,
#                                              skip_over_100k=True, log_degree=True, skip_snaps=False, normalize=False)

# run_parallel_local_degree_empirical_analysis(digg.egonet_files_path,
#                                              digg.directed_local_degree_empirical_base_results_path + 'directed-2/', 1,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=False, normalize=True)

# run_parallel_local_degree_empirical_analysis(digg.egonet_files_path,
#                                              digg.directed_local_degree_empirical_base_results_path + 'directed-3/', 1,
#                                              skip_over_100k=True, log_degree=True, skip_snaps=True, normalize=False)

# run_parallel_local_degree_empirical_analysis(digg.egonet_files_path,
#                                              digg.directed_local_degree_empirical_base_results_path + 'directed-4/', 3,
#                                              skip_over_100k=True, log_degree=False, skip_snaps=True, normalize=False)



# dgh.plot_local_degree_empirical_results(digg.directed_local_degree_empirical_base_results_path + 'directed-3/',
#                                         digg.directed_local_degree_empirical_base_results_path + 'directed-3/plots/',
#                                         gather_individual_results=False)

# dgh.plot_local_degree_empirical_cdf(digg.directed_local_degree_empirical_base_results_path + 'directed-3/',
#                                     digg.directed_local_degree_empirical_base_results_path + 'directed-3/plots/',
#                                     triangle_types=['T01'], separete_in_out_degree=True,
#                                     gather_individual_results=True)

# dgh.local_degree_empirical_result_comparison(digg.directed_local_degree_empirical_base_results_path + 'directed/',
#                                              include_conf_intervals=True, gather_individual_results=False)


# ************************************************************************* #
# ******************* Local Degree Distribution Analysis ****************** #
# ************************************************************************* #
def run_parallel_gather_local_degree_data(egonet_files_path, results_base_path, num_process, skip_over_100k,
                                          get_z_id_global_degree=False):
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)
        os.makedirs(results_base_path + 'results')
        os.makedirs(results_base_path + 'skipped_egonets')
        os.makedirs(results_base_path + 'temp-analyses-start')

    all_egonets = set(os.listdir(egonet_files_path))
    if skip_over_100k:
        analyzed_egonets = set(os.listdir(results_base_path + 'results')).union(os.listdir(results_base_path +
                                                                                           'skipped_egonets'))
    else:
        analyzed_egonets = set(os.listdir(results_base_path + 'results'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)
    np.random.shuffle(egonets_to_analyze)

    print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(dgh.gather_local_degree_data)
                                 (ego_net_file, results_base_path, egonet_files_path, get_z_id_global_degree,
                                  skip_over_100k) for ego_net_file in egonets_to_analyze)


# **** Google+ **** #
# run_parallel_gather_local_degree_data(gplus.egonet_files_path,
# #                                       gplus.local_degree_empirical_results_base_path +
# #                                       'local-degree-dist/pickle-files-1/', 6, skip_over_100k=True)

# run_parallel_gather_local_degree_data(gplus.egonet_files_path,
#                                       gplus.local_degree_empirical_results_base_path +
#                                       'local-degree-dist/pickle-files-2/', 6, skip_over_100k=True,
#                                       get_z_id_global_degree=True)



# dgh.plot_local_degree_distribution(gplus.local_degree_empirical_results_base_path + 'local-degree-dist/pickle-files-2/',
#                                    "", gather_individual_results=True)

# dgh.plot_local_degree_distribution(gplus.local_degree_empirical_results_base_path + 'local-degree-dist/pickle-files-2/',
#                                    "", gather_individual_results=False, get_z_id_global_degree=True)

# **** Flickr **** #
# run_parallel_gather_local_degree_data(flickr.egonet_files_path,
#                                       flickr.local_degree_dist_results_base_path + 'pickle-files-1/', 6,
#                                       skip_over_100k=True)

# run_parallel_gather_local_degree_data(flickr.egonet_files_path,
#                                       flickr.local_degree_dist_results_base_path + 'pickle-files-2/', 8,
#                                       skip_over_100k=True, get_z_id_global_degree=True)

# dgh.plot_local_degree_distribution(flickr.local_degree_dist_results_base_path + 'pickle-files-2/', "",
#                                    gather_individual_results=True)

# dgh.plot_local_degree_distribution(flickr.local_degree_dist_results_base_path + 'pickle-files-2/', "",
#                                    gather_individual_results=False, get_z_id_global_degree=True)

# **** Digg **** #
# run_parallel_gather_local_degree_data(digg.egonet_files_path,
#                                       digg.directed_local_degree_empirical_base_results_path +
#                                       'local-degree-dist/pickle-files-1/', 4, skip_over_100k=False)
# dgh.plot_local_degree_distribution(digg.directed_local_degree_empirical_base_results_path +
#                                    'local-degree-dist/pickle-files-1/',
#                                    digg.directed_local_degree_empirical_base_results_path +
#                                    'local-degree-dist/plots-1/', gather_individual_results=False)


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
# run_parallel_triad_ratio_analysis(gplus.egonet_files_path, gplus.triad_ratio_empirical_results_path_2, 6,
#                                   skip_over_100k=True)

# dgh.empirical_triad_list_formed_ratio_results_plot(gplus.triad_ratio_empirical_results_path_2,
#                                                    gplus.triad_ratio_empirical_plots_path_2,
#                                                    gather_individual_results=True)


# **** Flickr **** #
# run_parallel_triad_ratio_analysis(flickr.egonet_files_path, flickr.triad_ratio_empirical_results_path_2, 6,
#                                   skip_over_100k=True)

# dgh.empirical_triad_list_formed_ratio_results_plot(flickr.triad_ratio_empirical_results_path_2,
#                                                    flickr.triad_ratio_empirical_plots_path_2,
#                                                    gather_individual_results=True)


# **** Digg **** #
# run_parallel_triad_ratio_analysis(digg.egonet_files_path, digg.triad_ratio_empirical_results_path_2, 10,
#                                   skip_over_100k=True)

# dgh.empirical_triad_list_formed_ratio_results_plot(digg.triad_ratio_empirical_results_path_2,
#                                                    digg.triad_ratio_empirical_plots_path_2,
#                                                    gather_individual_results=True)


# ************************************************************************* #
# ************************ Link Prediction Analysis *********************** #
# ************************************************************************* #
def run_parallel_link_prediction_analysis(egonet_files_path, results_base_path, num_process, skip_over_100k,
                                          skip_snapshots_w_no_new_edge=True, separate_degree=False):
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)
        os.makedirs(results_base_path + 'analyzed_egonets')
        os.makedirs(results_base_path + 'cumulated_results')
        os.makedirs(results_base_path + 'results')
        os.makedirs(results_base_path + 'skipped_egonets')

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

    if not separate_degree:
        Parallel(n_jobs=num_process)(delayed(dgh.run_directed_link_prediction)
                                     (ego_net_file, top_k_values, egonet_files_path, results_base_path,
                                      skip_over_100k=skip_over_100k,
                                      skip_snapshots_w_no_new_edge=skip_snapshots_w_no_new_edge)
                                     for ego_net_file in egonets_to_analyze)
    else:
        Parallel(n_jobs=num_process)(delayed(dgh.run_directed_link_prediction_w_separate_degree)
                                     (ego_net_file, top_k_values, egonet_files_path, results_base_path,
                                      skip_over_100k=skip_over_100k,
                                      skip_snapshots_w_no_new_edge=skip_snapshots_w_no_new_edge)
                                     for ego_net_file in egonets_to_analyze)


# The methods to be compared for improvements plots
comparison_pairs = [('cn', 'dccn'), ('aa', 'dcaa')]
separate_deg_scores = ['od-dccn', 'in-dccn', 'od-aa', 'id-aa', 'od-dcaa', 'in-dcaa']
ld_no_log_scores = ['dccar', 'od-dccar', 'id-dccar', 'dccn', 'od-dccn', 'in-dccn', 'dcaa', 'od-dcaa', 'in-dcaa']

# **** Google+ **** #
# run_parallel_link_prediction_analysis(gplus.egonet_files_path, gplus.lp_results_path, 6, skip_over_100k=False)
# run_parallel_link_prediction_analysis(gplus.egonet_files_path, gplus.lp_results_base_path + 'pickle-files-2/', 4,
#                                       skip_over_100k=True, skip_snapshots_w_no_new_edge=False)
# run_parallel_link_prediction_analysis(gplus.egonet_files_path, gplus.lp_results_base_path + 'pickle-files-w-sep-deg/',
#                                       5, skip_over_100k=True, skip_snapshots_w_no_new_edge=False, separate_degree=True)
# run_parallel_link_prediction_analysis(gplus.egonet_files_path, gplus.lp_results_base_path + 'pickle-files-no-log/',
#                                       15, skip_over_100k=True, skip_snapshots_w_no_new_edge=False, separate_degree=True)

# lpe.calculate_lp_performance(gplus.lp_results_path, gather_individual_results=True)
# lpe.calculate_lp_performance(gplus.lp_results_base_path + 'pickle-files-2/', gather_individual_results=False)
# lpe.calculate_lp_performance(gplus.lp_results_base_path + 'pickle-files-w-sep-deg/', gather_individual_results=False,
#                              scores=separate_deg_scores)
# lpe.calculate_lp_performance(gplus.lp_results_base_path + 'pickle-files-no-log/', gather_individual_results=True,
#                              scores=ld_no_log_scores)

# lpe.plot_percent_improvements(gplus.lp_results_path, gplus.lp_plots_path, comparison_pairs,
#                               gather_individual_results=True)

# lpe.plot_lp_performance([gplus.lp_results_base_path + 'pickle-files-2/', gplus.lp_results_base_path + 'pickle-files-w-sep-deg/',
#                          gplus.lp_results_base_path + 'pickle-files-no-log/'],
#                         "/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots-2/", "Google+")
# lpe.plot_percent_improvements_all([gplus.lp_results_base_path + 'pickle-files-2/', gplus.lp_results_base_path + 'pickle-files-w-sep-deg/',
#                                    gplus.lp_results_base_path + 'pickle-files-no-log/'],
#                                   "/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots-2", "Google+")

# **** Flickr **** #
# run_parallel_link_prediction_analysis(flickr.egonet_files_path, flickr.lp_results_path, 6, skip_over_100k=False)
# run_parallel_link_prediction_analysis(flickr.egonet_files_path, flickr.lp_results_base_path + 'pickle-files-1/', 16,
#                                       skip_over_100k=True, skip_snapshots_w_no_new_edge=False)
# run_parallel_link_prediction_analysis(flickr.egonet_files_path, flickr.lp_results_base_path + 'pickle-files-w-sep-deg/',
#                                       12, skip_over_100k=True, skip_snapshots_w_no_new_edge=False, separate_degree=True)
# run_parallel_link_prediction_analysis(flickr.egonet_files_path, flickr.lp_results_base_path + 'pickle-files-no-log/',
#                                       12, skip_over_100k=True, skip_snapshots_w_no_new_edge=False, separate_degree=True)

# lpe.calculate_lp_performance(flickr.lp_results_path, gather_individual_results=True)
# lpe.calculate_lp_performance(flickr.lp_results_base_path + 'pickle-files-1/', gather_individual_results=False)
# lpe.calculate_lp_performance(flickr.lp_results_base_path + 'pickle-files-w-sep-deg/', gather_individual_results=False,
#                              scores=separate_deg_scores)
# lpe.calculate_lp_performance(flickr.lp_results_base_path + 'pickle-files-no-log/', gather_individual_results=True,
#                              scores=ld_no_log_scores)

# lpe.plot_percent_improvements(flickr.lp_results_path, flickr.lp_plots_path, comparison_pairs,
#                               gather_individual_results=False)

# lpe.plot_lp_performance([flickr.lp_results_base_path + 'pickle-files-1/', flickr.lp_results_base_path + 'pickle-files-w-sep-deg/', flickr.lp_results_base_path + 'pickle-files-no-log/'],
#                         "/shared/Results/EgocentricLinkPrediction/main/lp/flickr/plots/", "Flickr")
# lpe.plot_percent_improvements_all([flickr.lp_results_base_path + 'pickle-files-1/', flickr.lp_results_base_path + 'pickle-files-w-sep-deg/', flickr.lp_results_base_path + 'pickle-files-no-log/'],
#                         "/shared/Results/EgocentricLinkPrediction/main/lp/flickr/plots/", "Flickr")

# **** Digg **** #
# run_parallel_link_prediction_analysis(digg.egonet_files_path, digg.lp_results_file_path, 6, skip_over_100k=False)
# run_parallel_link_prediction_analysis(digg.egonet_files_path, digg.lp_results_file_base_path + 'pickle-files-1/', 6,
#                                       skip_over_100k=True, skip_snapshots_w_no_new_edge=False)
# run_parallel_link_prediction_analysis(digg.egonet_files_path, digg.lp_results_file_base_path + 'pickle-files-w-sep-deg/',
#                                       4, skip_over_100k=True, skip_snapshots_w_no_new_edge=False, separate_degree=True)
# run_parallel_link_prediction_analysis(digg.egonet_files_path, digg.lp_results_file_base_path + 'pickle-files-no-log/',
#                                       4, skip_over_100k=True, skip_snapshots_w_no_new_edge=False, separate_degree=True)

# lpe.calculate_lp_performance(digg.lp_results_file_path, gather_individual_results=True)
# lpe.calculate_lp_performance(digg.lp_results_file_base_path + 'pickle-files-1/', gather_individual_results=False)
# lpe.calculate_lp_performance(digg.lp_results_file_base_path + 'pickle-files-w-sep-deg/', gather_individual_results=False,
#                              scores=separate_deg_scores)
# lpe.calculate_lp_performance(digg.lp_results_file_base_path + 'pickle-files-no-log/', gather_individual_results=True,
#                              scores=ld_no_log_scores)

# lpe.plot_percent_improvements(digg.lp_results_file_path, digg.lp_plots_path, comparison_pairs,
#                               gather_individual_results=True)

# lpe.plot_lp_performance([digg.lp_results_file_base_path + 'pickle-files-1/', digg.lp_results_file_base_path + 'pickle-files-w-sep-deg/', digg.lp_results_file_base_path + 'pickle-files-no-log/'],
#                                   "/shared/Results/EgocentricLinkPrediction/main/lp/digg/directed/plots/", "Digg")
# lpe.plot_percent_improvements_all([digg.lp_results_file_base_path + 'pickle-files-1/', digg.lp_results_file_base_path + 'pickle-files-w-sep-deg/', digg.lp_results_file_base_path + 'pickle-files-no-log/'],
#                                   "/shared/Results/EgocentricLinkPrediction/main/lp/digg/directed/plots/", "Digg")

# ************************************************************************* #
# **************** Link Prediction Analysis on Test Methods *************** #
# ************************************************************************* #
def run_parallel_link_prediction_analysis_on_test_method(method_pointer, method_name, egonet_files_path,
                                                         results_base_path, num_process, skip_over_100k, num_samples=0,
                                                         specific_triads_only=False, wipe_older_results=False,
                                                         skip_snapshots_w_no_new_edge=True):

    if specific_triads_only:
        results_base_path = results_base_path + 'test-methods/specific-triads/'
        if not os.path.exists(results_base_path):
            os.makedirs(results_base_path)
    else:
        results_base_path = results_base_path + 'test-methods/'
    # Create directory if not exists
    if os.path.exists(results_base_path + method_name) and wipe_older_results:
        shutil.rmtree(results_base_path + method_name)

    if not os.path.exists(results_base_path + method_name):
        os.makedirs(results_base_path + method_name)
        os.makedirs(results_base_path + method_name + '/pickle-files')
        os.makedirs(results_base_path + method_name + '/pickle-files/analyzed_egonets')
        os.makedirs(results_base_path + method_name + '/pickle-files/results')
        os.makedirs(results_base_path + method_name + '/pickle-files/skipped_egonets')

    results_base_path = results_base_path + method_name + '/pickle-files/'

    top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
    all_egonets = set(os.listdir(egonet_files_path))
    if skip_over_100k:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets')).union(os.listdir(results_base_path +
                                                                                                    'skipped_egonets'))
    else:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)

    if num_samples == 0:
        np.random.shuffle(egonets_to_analyze)
    else:
        egonets_to_analyze = np.random.choice(egonets_to_analyze, size=num_samples, replace=False)

    print("{} egonets selected to analyze!".format(len(egonets_to_analyze)))

    if specific_triads_only:
        Parallel(n_jobs=num_process)(delayed(dgh.run_directed_link_prediction)
                                     (ego_net_file, top_k_values, egonet_files_path, results_base_path,
                                      specific_triads_only=specific_triads_only, skip_over_100k=skip_over_100k)
                                     for ego_net_file in egonets_to_analyze)

    else:
        Parallel(n_jobs=num_process)(delayed(dgh.run_directed_link_prediciton_on_test_method)
                                     (method_pointer, method_name, ego_net_file, top_k_values, egonet_files_path,
                                      results_base_path, skip_over_100k, skip_snapshots_w_no_new_edge)
                                     for ego_net_file in egonets_to_analyze)


# **** Google+ **** #
# run_parallel_link_prediction_analysis_on_test_method(dgh.test1_lp_scores_directed, 'test1', gplus.egonet_files_path,
#                                                      gplus.lp_results_base_path, num_samples=20, num_process=6,
#                                                      skip_over_100k=False, wipe_older_results=True)

# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['test1'], is_test=True,
#                              gather_individual_results=True)

# ** LD-CAR
# run_parallel_link_prediction_analysis_on_test_method(dgh.dccar_test1, 'dccar-test1', gplus.egonet_files_path,
#                                                      gplus.lp_results_base_path, num_samples=0, num_process=6,
#                                                      skip_over_100k=True, wipe_older_results=False)

# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['dccar-test1'], is_test=True,
#                              gather_individual_results=True)

# ** LD-CCLP
# run_parallel_link_prediction_analysis_on_test_method(dgh.dccclp_test1, 'dccclp-test1', gplus.egonet_files_path,
#                                                      gplus.lp_results_base_path, num_samples=0, num_process=6,
#                                                      skip_over_100k=True, wipe_older_results=False)

# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['dccclp-test1'], is_test=True,
#                              gather_individual_results=True)

# ** First Two Triads
# run_parallel_link_prediction_analysis_on_test_method('', 'first-two-triads', gplus.egonet_files_path,
#                                                      gplus.lp_results_base_path, num_process=6, skip_over_100k=True,
#                                                      num_samples=50000, specific_triads_only=True,
#                                                      wipe_older_results=False)

# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['first-two-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)

# ** First Three Triads with Five
# # run_parallel_link_prediction_analysis_on_test_method('', 'first-three-w-five-triads', gplus.egonet_files_path,
# #                                                      gplus.lp_results_base_path, num_process=6, skip_over_100k=True,
# #                                                      num_samples=50000, specific_triads_only=True,
# #                                                      wipe_older_results=False)
#
# # lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['first-three-w-five-triads'], is_test=False,
# #                              specific_triads_only=True, gather_individual_results=True)

# ** one, two, six, seven and nine
# run_parallel_link_prediction_analysis_on_test_method('', 'one-two-six-seven-nine', gplus.egonet_files_path,
#                                                      gplus.lp_results_base_path, num_process=10, skip_over_100k=True,
#                                                      num_samples=50000, specific_triads_only=True,
#                                                      wipe_older_results=False)

# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['one-two-six-seven-nine'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)


# ** Inverse Adamic Adar
# run_parallel_link_prediction_analysis_on_test_method(dgh.reverse_aa, 'reverse-aa', gplus.egonet_files_path,
#                                                      gplus.lp_results_base_path, num_samples=50000, num_process=4,
#                                                      skip_over_100k=True, wipe_older_results=False,
#                                                      skip_snapshots_w_no_new_edge=True)
#
# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['reverse-aa'], is_test=True,
#                              gather_individual_results=True)

# run_parallel_link_prediction_analysis_on_test_method(dgh.reverse_aa, 'reverse-aa-no-skip', gplus.egonet_files_path,
#                                                      gplus.lp_results_base_path, num_samples=50000, num_process=4,
#                                                      skip_over_100k=True, wipe_older_results=False,
#                                                      skip_snapshots_w_no_new_edge=False)

# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['reverse-aa-no-skip'], is_test=True,
#                              gather_individual_results=True)

# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['first-three-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)

# **** Flickr **** #
# run_parallel_link_prediction_analysis_on_test_method(dgh.test1_lp_scores_directed, 'test1', flickr.egonet_files_path,
#                                                      flickr.lp_results_base_path, num_samples=1000, num_process=6,
#                                                      skip_over_100k=False, wipe_older_results=False)

# lpe.calculate_lp_performance(flickr.lp_results_base_path, scores=['test1'], is_test=True,
#                              gather_individual_results=True)

# ** LD-CAR
# run_parallel_link_prediction_analysis_on_test_method(dgh.dccar_test1, 'dccar-test1', flickr.egonet_files_path,
#                                                      flickr.lp_results_base_path, num_samples=0, num_process=6,
#                                                      skip_over_100k=True, wipe_older_results=False)

# lpe.calculate_lp_performance(flickr.lp_results_base_path, scores=['dccar-test1'], is_test=True,
#                              gather_individual_results=True)


# ** LD-CCLP
# run_parallel_link_prediction_analysis_on_test_method(dgh.dccclp_test1, 'dccclp-test1', flickr.egonet_files_path,
#                                                      flickr.lp_results_base_path, num_samples=0, num_process=6,
#                                                      skip_over_100k=True, wipe_older_results=False)

# lpe.calculate_lp_performance(flickr.lp_results_base_path, scores=['dccclp-test1'], is_test=True,
#                              gather_individual_results=True)

# ** First Two Triads
# run_parallel_link_prediction_analysis_on_test_method('', 'first-two-triads', flickr.egonet_files_path,
#                                                      flickr.lp_results_base_path, num_process=6, skip_over_100k=True,
#                                                      num_samples=0, specific_triads_only=True,
#                                                      wipe_older_results=False)

# lpe.calculate_lp_performance(flickr.lp_results_base_path, scores=['first-two-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)

# ** First Three Triads with Five
# run_parallel_link_prediction_analysis_on_test_method('', 'first-three-w-five-triads', flickr.egonet_files_path,
#                                                      flickr.lp_results_base_path, num_process=10, skip_over_100k=True,
#                                                      num_samples=100000, specific_triads_only=True,
#                                                      wipe_older_results=False)

# lpe.calculate_lp_performance(flickr.lp_results_base_path, scores=['first-three-w-five-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)

# lpe.calculate_lp_performance(flickr.lp_results_base_path, scores=['first-three-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)

# **** Digg **** #
# run_parallel_link_prediction_analysis_on_test_method(dgh.test1_lp_scores_directed, 'test1', digg.egonet_files_path,
#                                                      digg.lp_results_file_base_path, num_samples=1000, num_process=6,
#                                                      skip_over_100k=False, wipe_older_results=False)

# lpe.calculate_lp_performance(digg.lp_results_file_base_path, scores=['test1'], is_test=True,
#                              gather_individual_results=True)

# LD-CAR
# run_parallel_link_prediction_analysis_on_test_method(dgh.dccar_test1, 'dccar-test1', digg.egonet_files_path,
#                                                      digg.lp_results_file_base_path, num_samples=0, num_process=20,
#                                                      skip_over_100k=False, wipe_older_results=False)

# lpe.calculate_lp_performance(digg.lp_results_file_base_path, scores=['dccar-test1'], is_test=True,
#                              gather_individual_results=True)

# LD-CCLP
# run_parallel_link_prediction_analysis_on_test_method(dgh.dccclp_test1, 'dccclp-test1', digg.egonet_files_path,
#                                                      digg.lp_results_file_base_path, num_samples=0, num_process=6,
#                                                      skip_over_100k=False, wipe_older_results=False)

# lpe.calculate_lp_performance(digg.lp_results_file_base_path, scores=['dccclp-test1'], is_test=True,
#                              gather_individual_results=True)


# ** First Two Triads
# run_parallel_link_prediction_analysis_on_test_method('', 'first-two-triads', digg.egonet_files_path,
#                                                      digg.lp_results_file_base_path, num_process=18,
#                                                      skip_over_100k=True, specific_triads_only=True,
#                                                      wipe_older_results=False)
#
# lpe.calculate_lp_performance(digg.lp_results_file_base_path, scores=['first-two-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)


# ** First Three Triads with Five
# run_parallel_link_prediction_analysis_on_test_method('', 'first-three-w-five-triads', digg.egonet_files_path,
#                                                      digg.lp_results_file_base_path, num_process=10,
#                                                      skip_over_100k=True, specific_triads_only=True,
#                                                      wipe_older_results=False)

# lpe.calculate_lp_performance(digg.lp_results_file_base_path, scores=['first-three-w-five-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)

# lpe.calculate_lp_performance(digg.lp_results_file_base_path, scores=['first-three-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=False)


# ******************************************************************************** #
# **************** Link Prediction Analysis on Personalized Triads *************** #
# ******************************************************************************** #
def run_parallel_link_prediction_analysis_on_personalized_triads(method_name, egonet_files_path, results_base_path,
                                                                 num_process, skip_over_100k, num_samples=0,
                                                                 wipe_older_results=False):

    results_base_path = results_base_path + 'test-methods/specific-triads/personalized/'
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)

    # Create directory if not exists
    if os.path.exists(results_base_path + method_name) and wipe_older_results:
        shutil.rmtree(results_base_path + method_name)

    if not os.path.exists(results_base_path + method_name):
        os.makedirs(results_base_path + method_name)
        os.makedirs(results_base_path + method_name + '/pickle-files')
        os.makedirs(results_base_path + method_name + '/pickle-files/analyzed_egonets')
        os.makedirs(results_base_path + method_name + '/pickle-files/results')
        os.makedirs(results_base_path + method_name + '/pickle-files/skipped_egonets')

    results_base_path = results_base_path + method_name + '/pickle-files/'

    top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
    all_egonets = set(os.listdir(egonet_files_path))
    if skip_over_100k:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets')).union(os.listdir(results_base_path +
                                                                                                    'skipped_egonets'))
    else:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)

    if num_samples == 0:
        np.random.shuffle(egonets_to_analyze)
    else:
        egonets_to_analyze = np.random.choice(egonets_to_analyze, size=num_samples, replace=False)

    print("{} egonets selected to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(dgh.run_directed_link_prediction_on_personalized_tirad)
                                 (ego_net_file, top_k_values, egonet_files_path, results_base_path,
                                  skip_over_100k=skip_over_100k)
                                 for ego_net_file in egonets_to_analyze)


# **** Google+ **** #
# run_parallel_link_prediction_analysis_on_personalized_triads('test1', gplus.egonet_files_path,
#                                                              gplus.lp_results_base_path, num_process=15,
#                                                              skip_over_100k=True, num_samples=50000,
#                                                              wipe_older_results=False)

# lpe.calculate_lp_performance_on_personalized_triads(gplus.lp_results_base_path, 'test1',
#                                                     gather_individual_results=True)

# **** Flickr **** #
# run_parallel_link_prediction_analysis_on_personalized_triads('test1', flickr.egonet_files_path,
#                                                              flickr.lp_results_base_path, num_process=6,
#                                                              skip_over_100k=True, num_samples=100000,
#                                                              wipe_older_results=False)

# lpe.calculate_lp_performance_on_personalized_triads(flickr.lp_results_base_path, 'test1',
#                                                     gather_individual_results=True)

# **** Digg **** #
# run_parallel_link_prediction_analysis_on_personalized_triads('test1', digg.egonet_files_path,
#                                                              digg.lp_results_file_base_path, num_process=6,
#                                                              skip_over_100k=True, wipe_older_results=False)

# lpe.calculate_lp_performance_on_personalized_triads(digg.lp_results_file_base_path, 'test1',
#                                                     gather_individual_results=True)


# ******************************************************************************** #
# **************** Link Prediction Analysis for Machine Learning ***************** #
# ******************************************************************************** #
def run_parallel_link_prediction_analysis_for_ml(egonet_files_path, results_base_path, num_process, skip_over_100k,
                                                 wipe_older_results=False):
    results_base_path = results_base_path + 'ml/'

    # wipe older data if needed
    if os.path.exists(results_base_path) and wipe_older_results:
        shutil.rmtree(results_base_path)

    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)
        os.makedirs(results_base_path + '/pickle-files')
        os.makedirs(results_base_path + '/pickle-files/analyzed_egonets')
        os.makedirs(results_base_path + '/pickle-files/results')
        os.makedirs(results_base_path + '/pickle-files/skipped_egonets')

    results_base_path = results_base_path + '/pickle-files/'

    all_egonets = set(os.listdir(egonet_files_path))
    if skip_over_100k:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets')).union(os.listdir(results_base_path +
                                                                                                    'skipped_egonets'))
    else:
        analyzed_egonets = set(os.listdir(results_base_path + 'analyzed_egonets'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)

    np.random.shuffle(egonets_to_analyze)

    print("{} egonets selected to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(dmlh.evaluate_lp_measures)
                                 (ego_net_file, egonet_files_path, results_base_path, skip_over_100k)
                                 for ego_net_file in egonets_to_analyze)


# **** Google+ **** #
# run_parallel_link_prediction_analysis_for_ml(gplus.egonet_files_path, gplus.lp_results_base_path, num_process=6,
#                                              skip_over_100k=True, wipe_older_results=False)
# dmlh.combine_all_egonet_data(gplus.lp_results_base_path + 'ml/pickle-files/', 2261949652)
# dmlh.combine_all_egonet_data(gplus.lp_results_base_path + 'ml/pickle-files/', num_samples=100000)
# dmlh.split_data_based_on_snapshot(gplus.lp_results_base_path + 'ml/pickle-files/', combined_file_name='random_20k.npy')


# dmlh.train_random_forest(gplus.lp_results_base_path + 'ml/pickle-files/', 'snapshot-0-random_20k.npy', '2', [2, 3, 4, 5, 6, 7, 8, 10, 11, 12])
# dmlh.train_random_forest(gplus.lp_results_base_path + 'ml/pickle-files/', 'snapshot-0-random_20k.npy', '3', [2, 3, 4, 5, 10, 11, 12], test_set_file_name='snapshot-1-random_20k.npy')
# dmlh.test_trained_model(gplus.lp_results_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-2.pickle", [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], "RF-2-snap-1-random-20k")
# dmlh.calc_percision_at_k(gplus.lp_results_base_path + 'ml/pickle-files/', "RF-2-snap-1-random-20k.pickle.npy")

# dmlh.test_trained_model(gplus.lp_results_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-3.pickle", [2, 3, 4, 5, 10, 11, 12], "RF-3-snap-1-random-20k")
# dmlh.calc_percision_at_k(gplus.lp_results_base_path + 'ml/pickle-files/', "RF-3-snap-1-random-20k.npy")

# dmlh.test_trained_model(gplus.lp_results_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-2.pickle")
# dmlh.aupr_trained_model(gplus.lp_results_base_path + 'ml/', 'snapshot-1-random_20k.npy', "RF-2.pickle", [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], "G+ Test 2")
# dmlh.aupr_trained_model(gplus.lp_results_base_path + 'ml/', 'snapshot-1-random_20k.npy', "RF-3.pickle", [2, 3, 4, 5, 10, 11, 12], "G+ Test 3")


# dmlh.logistic_reg_roller_learner(gplus.lp_results_base_path + 'ml/pickle-files/', 'random_20k.npy', 'roller-1', [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], n_jobs=6)
# dmlh.logistic_reg_roller_learner(gplus.lp_results_base_path + 'ml/pickle-files/', 'random_20k.npy', 'roller-2', [2, 3, 4, 5, 10, 11, 12], n_jobs=6)

# **** Flickr **** #
# run_parallel_link_prediction_analysis_for_ml(flickr.egonet_files_path, flickr.lp_results_base_path, num_process=7,
#                                              skip_over_100k=True, wipe_older_results=False)
# dmlh.combine_all_egonet_data(flickr.lp_results_base_path + 'ml/pickle-files/', 2407523156)
# dmlh.combine_all_egonet_data(flickr.lp_results_base_path + 'ml/pickle-files/', num_samples=100000)
# dmlh.split_data_based_on_snapshot(flickr.lp_results_base_path + 'ml/pickle-files/', combined_file_name='random_20k.npy')

# dmlh.train_random_forest(flickr.lp_results_base_path + 'ml/pickle-files/', 'snapshot-0-random_20k.npy', '2', [2, 3, 4, 5, 6, 7, 8, 10, 11, 12])
# dmlh.train_random_forest(flickr.lp_results_base_path + 'ml/pickle-files/', 'snapshot-0-random_20k.npy', '3', [2, 3, 4, 5, 10, 11, 12], test_set_file_name='snapshot-1-random_20k.npy')
# dmlh.test_trained_model(flickr.lp_results_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-2.pickle")
# dmlh.aupr_trained_model(flickr.lp_results_base_path + 'ml/', 'snapshot-1-random_20k.npy', "RF-2.pickle", [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], "Flickr Test 2")
# dmlh.aupr_trained_model(flickr.lp_results_base_path + 'ml/', 'snapshot-1-random_20k.npy', "RF-3.pickle", [2, 3, 4, 5, 10, 11, 12], "Flickr Test 3")

# dmlh.test_trained_model(flickr.lp_results_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-2.pickle", [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], "RF-2-snap-1-random-20k")
# dmlh.calc_percision_at_k(flickr.lp_results_base_path + 'ml/pickle-files/', "RF-2-snap-1-random-20k.npy")
#
# dmlh.test_trained_model(flickr.lp_results_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-3.pickle", [2, 3, 4, 5, 10, 11, 12], "RF-3-snap-1-random-20k")
# dmlh.calc_percision_at_k(flickr.lp_results_base_path + 'ml/pickle-files/', "RF-3-snap-1-random-20k.npy")


# dmlh.logistic_reg_roller_learner(flickr.lp_results_base_path + 'ml/pickle-files/', 'random_20k.npy', 'roller-1', [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], n_jobs=12)
# dmlh.logistic_reg_roller_learner(flickr.lp_results_base_path + 'ml/pickle-files/', 'random_20k.npy', 'roller-2', [2, 3, 4, 5, 10, 11, 12], n_jobs=12)

# **** Digg **** #
# run_parallel_link_prediction_analysis_for_ml(digg.egonet_files_path, digg.lp_results_file_base_path, num_process=20,
#                                              skip_over_100k=True, wipe_older_results=False)
# dmlh.combine_all_egonet_data(digg.lp_results_file_base_path + 'ml/pickle-files/', 875298799)
# dmlh.combine_all_egonet_data(digg.lp_results_file_base_path + 'ml/pickle-files/')
# dmlh.split_data_based_on_snapshot(digg.lp_results_file_base_path + 'ml/pickle-files/', combined_file_name='random_20k.npy')

# dmlh.train_random_forest(digg.lp_results_file_base_path + 'ml/pickle-files/', 'snapshot-0-random_20k.npy', '2', [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], n_jobs=22)
# dmlh.train_random_forest(digg.lp_results_file_base_path + 'ml/pickle-files/', 'snapshot-0-random_20k.npy', '3', [2, 3, 4, 5, 10, 11, 12], test_set_file_name='snapshot-1-random_20k.npy', n_jobs=22)
# dmlh.test_trained_model(digg.lp_results_file_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-2.pickle")
# dmlh.aupr_trained_model(digg.lp_results_file_base_path + 'ml/', 'snapshot-1-random_20k.npy', "RF-2.pickle", [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], "Digg Test 2")
# dmlh.aupr_trained_model(digg.lp_results_file_base_path + 'ml/', 'snapshot-1-random_20k.npy', "RF-3.pickle", [2, 3, 4, 5, 10, 11, 12], "Digg Test 3")

# dmlh.test_trained_model(digg.lp_results_file_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-2.pickle", [2, 3, 4, 5, 6, 7, 8, 10, 11, 12], "RF-2-snap-1-random-20k")
# dmlh.calc_percision_at_k(digg.lp_results_file_base_path + 'ml/pickle-files/', "RF-2-snap-1-random-20k.npy")
#
# dmlh.test_trained_model(digg.lp_results_file_base_path + 'ml/pickle-files/', 'snapshot-1-random_20k.npy', "RF-3.pickle", [2, 3, 4, 5, 10, 11, 12], "RF-3-snap-1-random-20k")
# dmlh.calc_percision_at_k(digg.lp_results_file_base_path + 'ml/pickle-files/', "RF-3-snap-1-random-20k.npy")


# dmlh.logistic_reg_roller_learner(digg.lp_results_file_base_path + 'ml/pickle-files/', 'random_20k.npy', 'roller-1', [2, 3, 4, 5, 6, 7, 8, 10, 11, 12])
# dmlh.logistic_reg_roller_learner(digg.lp_results_file_base_path + 'ml/pickle-files/', 'random_20k.npy', 'roller-2', [2, 3, 4, 5, 10, 11, 12])