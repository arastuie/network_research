import os
import numpy as np
import facebook_helpers as fb
from joblib import Parallel, delayed
import link_prediction_helpers as lph


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
# Parallel(n_jobs=23)(delayed(fb.run_link_prediction_analysis)(ego_net_file, top_k_values)
#                     for ego_net_file in egonets_to_analyze)

# LP Plot
lph.calculate_lp_performance(fb.lp_results_path + 'before-pymk/', gather_individual_lp_results=True)
lph.calculate_lp_performance(fb.lp_results_path + 'after-pymk/', gather_individual_lp_results=True)
