import os
import numpy as np
import facebook_helpers as fb
from joblib import Parallel, delayed


# ************************************************************************* #
# ******************** Local Degree Empirical Analysis ******************** #
# ************************************************************************* #

# **** Facebook **** #
# all_egonets = set(os.listdir(fb.egonet_files_path))
# analyzed_egonets = set(os.listdir(fb.empirical_pickle_path + 'after-pymk/analyzed-egonets'))
#
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=24)(delayed(fb.run_local_degree_empirical_analysis)(ego_net_file) for ego_net_file in egonets_to_analyze)

# Plotting
# fb.plot_local_degree_empirical_results(gather_individual_results=False)
# fb.plot_local_degree_empirical_ecdf(gather_individual_results=True)

# ************************************************************************* #
# ************************ Link Prediction Analysis *********************** #
# ************************************************************************* #

# **** Facebook **** #
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
# all_egonets = set(os.listdir(fb.egonet_files_path))
# analyzed_egonets = set(os.listdir(fb.lp_results_path + 'after-pymk/analyzed_egonets'))
#
# egonets_to_analyze = list(all_egonets - analyzed_egonets)
# np.random.shuffle(egonets_to_analyze)
#
# print("{} egonets left to analyze!".format(len(egonets_to_analyze)))
#
# Parallel(n_jobs=6)(delayed(fb.run_link_prediction_analysis)(ego_net_file, top_k_values)
#                    for ego_net_file in egonets_to_analyze)

# Result Calculation
score_list = ['cn', 'dccn', 'aa', 'dcaa', 'car', 'dccar', 'cclp', 'dccclp']
fb.calculate_lp_performance(scores=score_list, gather_individual_results=True)

# comparison_pairs = [('cn', 'dccn'), ('aa', 'dcaa')]
# fb.plot_percent_improvements(comparison_pairs, gather_individual_results=False)
