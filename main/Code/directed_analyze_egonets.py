import os
import shutil
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

# dgh.plot_local_degree_empirical_results(gplus.local_degree_empirical_results_path,
#                                         gplus.local_degree_empirical_plot_path, gather_individual_results=False)

# dgh.plot_local_degree_empirical_cdf(gplus.local_degree_empirical_results_path, gplus.local_degree_empirical_plot_path,
#                                     triangle_types='all', separete_in_out_degree=False,
#                                     gather_individual_results=True)

# dgh.local_degree_empirical_result_comparison(gplus.local_degree_empirical_results_path, include_conf_intervals=False,
#                                              gather_individual_results=False)


# **** Flickr **** #
# run_parallel_local_degree_empirical_analysis(flickr.egonet_files_path, flickr.local_degree_empirical_results_path, 6,
#                                              skip_over_100k=False)

# dgh.plot_local_degree_empirical_results(flickr.local_degree_empirical_results_path,
#                                         flickr.local_degree_empirical_plot_path, gather_individual_results=True)

# dgh.plot_local_degree_empirical_cdf(flickr.local_degree_empirical_results_path,
#                                     flickr.local_degree_empirical_plot_path, triangle_types='all',
#                                     separete_in_out_degree=False, gather_individual_results=True)

# dgh.local_degree_empirical_result_comparison(flickr.local_degree_empirical_results_path, include_conf_intervals=False,
#                                              gather_individual_results=False)


# **** Digg **** #
# run_parallel_local_degree_empirical_analysis(digg.egonet_files_path,
#                                              digg.directed_local_degree_empirical_results_path, 17,
#                                              skip_over_100k=False)

# dgh.plot_local_degree_empirical_results(digg.directed_local_degree_empirical_results_path,
#                                         digg.directed_local_degree_empirical_plot_path,
#                                         gather_individual_results=True)

# dgh.plot_local_degree_empirical_cdf(digg.directed_local_degree_empirical_results_path,
#                                     digg.directed_local_degree_empirical_plot_path, triangle_types='all',
#                                     separete_in_out_degree=False, gather_individual_results=True)

# dgh.local_degree_empirical_result_comparison(digg.directed_local_degree_empirical_results_path,
#                                              include_conf_intervals=False, gather_individual_results=False)


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
# run_parallel_triad_ratio_analysis(gplus.egonet_files_path, gplus.triad_ratio_empirical_results_path_1, 6,
#                                   skip_over_100k=True)

# dgh.empirical_triad_list_formed_ratio_results_plot(gplus.triad_ratio_empirical_results_path_1,
#                                                    gplus.triad_ratio_empirical_plots_path_1,
#                                                    gather_individual_results=True)


# **** Flickr **** #
# run_parallel_triad_ratio_analysis(flickr.egonet_files_path, flickr.triad_ratio_empirical_results_path_1, 6,
#                                   skip_over_100k=True)

# dgh.empirical_triad_list_formed_ratio_results_plot(flickr.triad_ratio_empirical_results_path_1,
#                                                    flickr.triad_ratio_empirical_plots_path_1,
#                                                    gather_individual_results=True)


# **** Digg **** #
run_parallel_triad_ratio_analysis(digg.egonet_files_path, digg.triad_ratio_empirical_results_path_1, 20,
                                  skip_over_100k=True)

# dgh.empirical_triad_list_formed_ratio_results_plot(digg.triad_ratio_empirical_results_path_1,
#                                                    digg.triad_ratio_empirical_plots_path_1,
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
                                 (ego_net_file, top_k_values, egonet_files_path, results_base_path,
                                  skip_over_100k=skip_over_100k) for ego_net_file in egonets_to_analyze)


# The methods to be compared for improvements plots
comparison_pairs = [('cn', 'dccn'), ('aa', 'dcaa')]

# **** Google+ **** #
# run_parallel_link_prediction_analysis(gplus.egonet_files_path, gplus.lp_results_path, 6, skip_over_100k=False)

# lpe.calculate_lp_performance(gplus.lp_results_path, gather_individual_results=True)

# lpe.plot_percent_improvements(gplus.lp_results_path, gplus.lp_plots_path, comparison_pairs,
#                               gather_individual_results=True)


# **** Flickr **** #
# run_parallel_link_prediction_analysis(flickr.egonet_files_path, flickr.lp_results_path, 6, skip_over_100k=False)

# lpe.calculate_lp_performance(flickr.lp_results_path, gather_individual_results=True)

# lpe.plot_percent_improvements(flickr.lp_results_path, flickr.lp_plots_path, comparison_pairs,
#                               gather_individual_results=False)


# **** Digg **** #
# run_parallel_link_prediction_analysis(digg.egonet_files_path, digg.lp_results_file_path, 6, skip_over_100k=False)

# lpe.calculate_lp_performance(digg.lp_results_file_path, gather_individual_results=True)

# lpe.plot_percent_improvements(digg.lp_results_file_path, digg.lp_plots_path, comparison_pairs,
#                               gather_individual_results=True)


# ************************************************************************* #
# **************** Link Prediction Analysis on Test Methods *************** #
# ************************************************************************* #
def run_parallel_link_prediction_analysis_on_test_method(method_pointer, method_name, egonet_files_path,
                                                         results_base_path, num_process, skip_over_100k, num_samples=0,
                                                         specific_triads_only=False, wipe_older_results=False):

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
                                      results_base_path, skip_over_100k) for ego_net_file in egonets_to_analyze)


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
# run_parallel_link_prediction_analysis_on_test_method('', 'first-three-w-five-triads', gplus.egonet_files_path,
#                                                      gplus.lp_results_base_path, num_process=6, skip_over_100k=True,
#                                                      num_samples=50000, specific_triads_only=True,
#                                                      wipe_older_results=False)

# lpe.calculate_lp_performance(gplus.lp_results_base_path, scores=['first-three-w-five-triads'], is_test=False,
#                              specific_triads_only=True, gather_individual_results=True)


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

lpe.calculate_lp_performance(flickr.lp_results_base_path, scores=['dccclp-test1'], is_test=True,
                             gather_individual_results=True)

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
