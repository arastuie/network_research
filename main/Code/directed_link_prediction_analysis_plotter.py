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


def plot_lp_errorbar(score_1, err_1, label_1, score_2, err_2, label_2, top_k, trig_type, plot_label, plot_path):
    plt.figure()
    plt.rc('legend', fontsize=17)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.errorbar(top_k, score_1, yerr=err_1, marker='o', color='b', ecolor='r', elinewidth=2, label=label_1)
    plt.errorbar(top_k, score_2, yerr=err_2, marker='^', color='g', ecolor='y', elinewidth=2, label=label_2)

    if trig_type != '':
        plt.title(trig_type, fontsize=17)

    plt.ylabel(plot_label, fontsize=15)
    plt.xlabel('Top K Value', fontsize=15)
    plt.legend(loc='lower right')
    plt.show()
    # current_fig = plt.gcf()
    # current_fig.savefig(plot_path, format='pdf')
    plt.clf()


# # LP results pickle files are in the following order
# #   cn, dccn_i, dccn_o, aa_i, aa_o, dcaa_i, dcaa_o
#
#
# score_list = ['cn', 'dccn_i', 'dccn_o', 'aa_i', 'aa_o', 'dcaa_i', 'dcaa_o']
# triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/'
# plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
# # # Computing percent improvements and standard errors over all the ego nets analyzed
# for triangle_type in triangle_types:
#     percent_imp_cn_i = {}
#     percent_imp_cn_o = {}
#     percent_imp_aa_i = {}
#     percent_imp_aa_o = {}
#
#     for k in top_k_values:
#         percent_imp_cn_i[k] = []
#         percent_imp_cn_o[k] = []
#         percent_imp_aa_i[k] = []
#         percent_imp_aa_o[k] = []
#
#     # loading result data
#     for result_file in os.listdir(result_file_base_path + triangle_type):
#         with open(result_file_base_path + triangle_type + '/' + result_file, 'rb') as f:
#             egonet_lp_results = pickle.load(f)
#
#         for k in top_k_values:
#             calc_percent_imp(percent_imp_cn_i, egonet_lp_results, 'cn', 'dccn_i', k)
#             calc_percent_imp(percent_imp_cn_o, egonet_lp_results, 'cn', 'dccn_o', k)
#             calc_percent_imp(percent_imp_aa_i, egonet_lp_results, 'aa_i', 'dcaa_i', k)
#             calc_percent_imp(percent_imp_aa_o, egonet_lp_results, 'aa_o', 'dcaa_o', k)
#
#     imp_mse = {
#         'imp_cn_i': [],
#         'imp_cn_i_err': [],
#         'imp_cn_o': [],
#         'imp_cn_o_err': [],
#         'imp_aa_i': [],
#         'imp_aa_i_err': [],
#         'imp_aa_o': [],
#         'imp_aa_o_err': []
#     }
#
#     for k in top_k_values:
#         imp_mse['imp_cn_i'].append(np.mean(percent_imp_cn_i[k]) * 100)
#         imp_mse['imp_cn_i_err'].append(np.std(percent_imp_cn_i[k]) / np.sqrt(len(percent_imp_cn_i[k])) * 100)
#
#         imp_mse['imp_cn_o'].append(np.mean(percent_imp_cn_o[k]) * 100)
#         imp_mse['imp_cn_o_err'].append(np.std(percent_imp_cn_o[k]) / np.sqrt(len(percent_imp_cn_o[k])) * 100)
#
#         imp_mse['imp_aa_i'].append(np.mean(percent_imp_aa_i[k]) * 100)
#         imp_mse['imp_aa_i_err'].append(np.std(percent_imp_aa_i[k]) / np.sqrt(len(percent_imp_aa_i[k])) * 100)
#
#         imp_mse['imp_aa_o'].append(np.mean(percent_imp_aa_o[k]) * 100)
#         imp_mse['imp_aa_o_err'].append(np.std(percent_imp_aa_o[k]) / np.sqrt(len(percent_imp_aa_o[k])) * 100)
#
#     with open(result_file_base_path + 'plot_ready_data/' + triangle_type + '.pckle', 'wb') as f:
#         pickle.dump(imp_mse, f, protocol=-1)
#
#     print(triangle_type + ": Done")
#
# print("Percent improvement calculation Done!")
#
# # plotting
# for triangle_type in triangle_types:
#     with open(result_file_base_path + 'plot_ready_data/' + triangle_type + '.pckle', 'rb') as f:
#         imp_mse = pickle.load(f)
#
#     plot_lp_errorbar(imp_mse['imp_cn_i'], imp_mse['imp_cn_i_err'], 'Indgree',
#                      imp_mse['imp_cn_o'],  imp_mse['imp_cn_o_err'], 'Outdegree', top_k_values, triangle_type,
#                      'DCCN vs CN Percent Improvement', '{0}/{1}-dccn-cn.pdf'.format(plot_save_path, triangle_type))
#
#     plot_lp_errorbar(imp_mse['imp_aa_i'], imp_mse['imp_aa_i_err'], 'Indgree',
#                      imp_mse['imp_aa_o'],  imp_mse['imp_aa_o_err'], 'Outdegree', top_k_values, triangle_type,
#                      'DCAA vs AA Percent Improvement', '{0}/{1}-dcaa-aa.pdf'.format(plot_save_path, triangle_type))
#
#     print(triangle_type + ": Done")
#
# print("Plotting is Done!")



##################### COMBINED #####################
# LP results pickle files are in the following order
#   cn, dccn, aa, dcaa

# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/'
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/test-2/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]

# # Computing percent improvements and standard errors over all the ego nets analyzed
percent_imp_cn = {}
percent_imp_aa = {}

for k in top_k_values:
    percent_imp_cn[k] = []
    percent_imp_aa[k] = []

# loading result data
for result_file in os.listdir(result_file_base_path + 'results'):
    with open(result_file_base_path + 'results/' + result_file, 'rb') as f:
        egonet_lp_results = pickle.load(f)

    for k in top_k_values:
        calc_percent_imp(percent_imp_cn, egonet_lp_results, 'cn', 'dccn', k)
        calc_percent_imp(percent_imp_aa, egonet_lp_results, 'aa', 'dcaa', k)

imp_mse = {
    'imp_cn': [],
    'imp_cn_err': [],
    'imp_aa': [],
    'imp_aa_err': [],
}

for k in top_k_values:
    imp_mse['imp_cn'].append(np.mean(percent_imp_cn[k]) * 100)
    imp_mse['imp_cn_err'].append(np.std(percent_imp_cn[k]) / np.sqrt(len(percent_imp_cn[k])) * 100)

    imp_mse['imp_aa'].append(np.mean(percent_imp_aa[k]) * 100)
    imp_mse['imp_aa_err'].append(np.std(percent_imp_aa[k]) / np.sqrt(len(percent_imp_aa[k])) * 100)

# with open(result_file_base_path + 'plot_ready_data/all-results.pckle', 'wb') as f:
#     pickle.dump(imp_mse, f, protocol=-1)

# print("Result collection and calculation: Done")
#
# # plotting
# with open(result_file_base_path + 'plot_ready_data/all-results.pckle', 'rb') as f:
#     imp_mse = pickle.load(f)

plot_lp_errorbar(imp_mse['imp_cn'], imp_mse['imp_cn_err'], 'DCCN VS CN',
                 imp_mse['imp_aa'],  imp_mse['imp_aa_err'], 'DCAA VS AA', top_k_values, '',
                 'Degree Corrected Percent Improvement', '{0}/gplus-lp-combined-test-2.pdf'.format(plot_save_path))

print("Plotting is Done!")