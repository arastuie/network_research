import os
import numpy as np
import facebook_helpers as fb
from joblib import Parallel, delayed


# ************************************************************************* #
# ******************** Local Degree Empirical Analysis ******************** #
# ************************************************************************* #
def run_parallel_local_degree_empirical_analysis(results_base_path, num_process, log_degree,
                                                 skip_snaps, normalize):
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)
        for path in ['before-pymk', 'after-pymk']:
            os.makedirs(results_base_path + path)
            os.makedirs(results_base_path + path + '/all-scores')
            os.makedirs(results_base_path + path + '/analyzed-egonets')
            os.makedirs(results_base_path + path + '/results')

    all_egonets = set(os.listdir(fb.egonet_files_path))
    analyzed_egonets = set(os.listdir(results_base_path + 'after-pymk/analyzed-egonets'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)
    np.random.shuffle(egonets_to_analyze)

    print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(fb.run_local_degree_empirical_analysis)
                                 (ego_net_file, results_base_path, log_degree, skip_snaps, normalize)
                                 for ego_net_file in egonets_to_analyze)

# **** Facebook **** #
# run_parallel_local_degree_empirical_analysis(fb.empirical_pickle_base_path + 'pickle-files-1/', 1,
#                                              log_degree=False, skip_snaps=True, normalize=True)

# run_parallel_local_degree_empirical_analysis(fb.empirical_pickle_base_path + 'pickle-files-2/', 1,
#                                              log_degree=True, skip_snaps=False, normalize=False)

# run_parallel_local_degree_empirical_analysis(fb.empirical_pickle_base_path + 'pickle-files-3/', 1,
#                                              log_degree=False, skip_snaps=False, normalize=True)

# run_parallel_local_degree_empirical_analysis(fb.empirical_pickle_base_path + 'pickle-files-4/', 1,
#                                              log_degree=True, skip_snaps=True, normalize=False)

# run_parallel_local_degree_empirical_analysis(fb.empirical_pickle_base_path + 'pickle-files-5/', 3,
#                                              log_degree=False, skip_snaps=True, normalize=False)


# Plotting
# fb.plot_local_degree_empirical_results(fb.empirical_pickle_base_path + 'pickle-files-4/',
#                                        fb.empirical_pickle_base_path + 'plots-4/', gather_individual_results=False)

# fb.plot_local_degree_empirical_ecdf(fb.empirical_pickle_base_path + 'pickle-files-4/',
#                                     fb.empirical_pickle_base_path + 'plots-4/', gather_individual_results=True)


# ************************************************************************* #
# ****************** Local Degree Distribution Analysis ******************* #
# ************************************************************************* #
def run_parallel_local_degree_distribution_gathering(results_base_path, num_process):
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)
        for path in ['before-pymk', 'after-pymk']:
            os.makedirs(results_base_path + path)
            os.makedirs(results_base_path + path + '/results')

    all_egonets = set(os.listdir(fb.egonet_files_path))
    analyzed_egonets = set(os.listdir(results_base_path + 'after-pymk/results'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)
    np.random.shuffle(egonets_to_analyze)

    print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(fb.gather_local_degree_data)(ego_net_file, results_base_path)
                                 for ego_net_file in egonets_to_analyze)


# run_parallel_local_degree_distribution_gathering(fb.empirical_pickle_base_path + 'local-degree-dist/pickle-files-1/',
#                                                  4)

# fb.plot_local_degree_distribution_over_single_ego(fb.empirical_pickle_base_path + 'local-degree-dist/pickle-files-1/',
#                                                   "", 200)

# fb.plot_local_degree_distribution(fb.empirical_pickle_base_path + 'local-degree-dist/pickle-files-1/',
#                                   fb.empirical_pickle_base_path + 'local-degree-dist/plots/', gather_individual_results=False)


# ************************************************************************* #
# ************************ Link Prediction Analysis *********************** #
# ************************************************************************* #
def run_parallel_link_prediction_analysis(results_base_path, num_process):
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)
        for path in ['before-pymk', 'after-pymk']:
            os.makedirs(results_base_path + path)
            os.makedirs(results_base_path + path + '/cumulated_results')
            os.makedirs(results_base_path + path + '/analyzed_egonets')
            os.makedirs(results_base_path + path + '/results')

    top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
    all_egonets = set(os.listdir(fb.egonet_files_path))
    analyzed_egonets = set(os.listdir(results_base_path + 'after-pymk/analyzed_egonets'))

    egonets_to_analyze = list(all_egonets - analyzed_egonets)
    np.random.shuffle(egonets_to_analyze)

    print("{} egonets left to analyze!".format(len(egonets_to_analyze)))

    Parallel(n_jobs=num_process)(delayed(fb.run_link_prediction_analysis)(ego_net_file, results_base_path, top_k_values)
                                 for ego_net_file in egonets_to_analyze)


# **** Facebook **** #
# run_parallel_link_prediction_analysis(fb.lp_results_path, 6)
# run_parallel_link_prediction_analysis('/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files-3/', 24)

# Result Calculation
# score_list = ['dccn', 'dccar', 'dcaa']
# fb.calculate_lp_performance('/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files-3/', scores=score_list,
#                             togather_individual_results=True)

# score_list = ['cn', 'dccn', 'aa', 'dcaa', 'car', 'dccar', 'cclp', 'dccclp']
# fb.calculate_lp_performance(fb.lp_results_path, scores=score_list, gather_individual_results=True)

# comparison_pairs = [('cn', 'dccn'), ('aa', 'dcaa')]
# fb.plot_percent_improvements(comparison_pairs, gather_individual_results=False)

fb.plot_lp_performance_bar_plot('/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files-2/')
fb.plot_percent_improvements_all('/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files-2/')