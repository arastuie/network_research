import os
import math
import pickle
import numpy as np
import helpers as h
import matplotlib.pyplot as plt


def calc_percent_imp(percent_imp_list, results, base_score, improved_score, k):
    if results[base_score][k] != 0:
        percent_imp_list[k].append((results[improved_score][k] - results[base_score][k]) / results[base_score][k])
    else:
        percent_imp_list[k].append(results[improved_score][k])

# LP results pickle files are in the following order
#   cn, dccn_i, dccn_o, aa_i, aa_o, dcaa_i, dcaa_o

score_list = ['cn', 'dccn_i', 'dccn_o', 'aa_i', 'aa_o', 'dcaa_i', 'dcaa_o']
triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]


for triangle_type in triangle_types:
    percent_imp_aa_i = {}
    percent_imp_aa_o = {}
    percent_imp_cn_i = {}
    percent_imp_cn_o = {}

    for k in top_k_values:
        percent_imp_aa_i[k] = []
        percent_imp_aa_o[k] = []
        percent_imp_cn_i[k] = []
        percent_imp_cn_o[k] = []

    # loading result data
    for result_file in os.listdir(result_file_base_path + triangle_type):
        with open(result_file_base_path + triangle_type + '/' + result_file, 'rb') as f:
            egonet_lp_results = pickle.load(f)

        for k in top_k_values:
            if egonet_lp_results['cn'][k] != 0:
                percent_imp_cn_i[k].append((egonet_lp_results['dccn_i'][k] - egonet_lp_results['cn'][k]) / egonet_lp_results['cn'][k])
            else:
                percent_imp_cn_i[k].append(egonet_lp_results['dccn_i'][k])

            if percent_aa[k][i] != 0:
                percent_imp_aa_i[k].append((percent_dcaa[k][i] - percent_aa[k][i]) / percent_aa[k][i])
            else:
                percent_imp_aa[k].append(percent_dcaa[k][i])

            if percent_cn[k][i] != 0:
                percent_imp_cn[k].append((percent_dccn[k][i] - percent_cn[k][i]) / percent_cn[k][i])
            else:
                percent_imp_cn[k].append(percent_dccn[k][i])

        with open('{0}/{1}/temp/total-result-percent-imp.pckl'.format(path, p), 'wb') as f:
            pickle.dump([percent_imp_aa, percent_imp_cn], f, protocol=-1)


    print(triangle_type + ": Done")

print("Done!")
