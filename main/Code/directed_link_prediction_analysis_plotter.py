import os
import math
import pickle
import numpy as np
import helpers as h
import matplotlib.pyplot as plt


def calc_percent_imp(percent_imp_list, results, base_score, improved_score, ki):
    if results[base_score][ki] != 0:
        percent_imp_list[ki].append((results[improved_score][ki] - results[base_score][ki]) / results[base_score][ki])
    else:
        percent_imp_list[ki].append(results[improved_score][ki])

# LP results pickle files are in the following order
#   cn, dccn_i, dccn_o, aa_i, aa_o, dcaa_i, dcaa_o


score_list = ['cn', 'dccn_i', 'dccn_o', 'aa_i', 'aa_o', 'dcaa_i', 'dcaa_o']
triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]

# Computing percent improvements and standard errors over all the ego nets analyzed
for triangle_type in triangle_types:
    percent_imp_cn_i = {}
    percent_imp_cn_o = {}
    percent_imp_aa_i = {}
    percent_imp_aa_o = {}

    for k in top_k_values:
        percent_imp_cn_i[k] = []
        percent_imp_cn_o[k] = []
        percent_imp_aa_i[k] = []
        percent_imp_aa_o[k] = []

    # loading result data
    for result_file in os.listdir(result_file_base_path + triangle_type):
        with open(result_file_base_path + triangle_type + '/' + result_file, 'rb') as f:
            egonet_lp_results = pickle.load(f)

        for k in top_k_values:
            calc_percent_imp(percent_imp_cn_i, egonet_lp_results, 'cn', 'dccn_i', k)
            calc_percent_imp(percent_imp_cn_o, egonet_lp_results, 'cn', 'dccn_o', k)
            calc_percent_imp(percent_imp_aa_i, egonet_lp_results, 'aa_i', 'dcaa_i', k)
            calc_percent_imp(percent_imp_aa_o, egonet_lp_results, 'aa_o', 'dcaa_o', k)


    with open(result_file_base_path + 'percent_imp_scores/' + triangle_type + '.pckle', 'wb') as f:
        pickle.dump([percent_imp_cn_i, percent_imp_cn_o, percent_imp_aa_i, percent_imp_aa_o], f, protocol=-1)

    print(triangle_type + ": Done")

print("Percent improvement calculation Done!")


# for triangle_type in triangle_types:
