import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st


def get_conf(list):
    m = np.mean(list) * 100
    # err = 100 * np.std(list) / np.sqrt(len(list))
    #
    # up = round(m + err, 2)
    # down = round(m - err, 2)
    # return "({0}, {1})".format(down, up)
    return "{0}".format(round(m, 2))


path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]

paths = ['lower-6', 'after-6']

lower_imp = []
lower_imp_err = []
after_imp = []
after_imp_err = []
for p in paths:
    percent_imp = {}

    for k in top_k_values:
        percent_imp[k] = []

    with open('{0}/pickle-files/{1}/temp-3/total-result.pckl'.format(path, p), 'rb') as f:
        percent_aa, percent_dcaa, percent_cn, percent_dccn = pickle.load(f)

        for k in top_k_values:
            for i in range(len(percent_cn[k])):
                if percent_cn[k][i] != 0:
                    percent_imp[k].append((percent_aa[k][i] - percent_cn[k][i]) / percent_cn[k][i])
                else:
                    percent_imp[k].append(percent_aa[k][i])

    if p == 'lower-6':
        for k in top_k_values:
            lower_imp.append(np.mean(percent_imp[k]) * 100)
            lower_imp_err.append(np.std(percent_imp[k]) / np.sqrt(len(percent_imp[k])) * 100)

        print("Before PYMK:")

        print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")

        print("CN: ", end=', \t')
        for k in top_k_values:
            print(get_conf(percent_cn[k]), end='& ')
        print("")

        print("DCCN: ", end=', \t')
        for k in top_k_values:
            print(get_conf(percent_dccn[k]), end='& ')
        print("")

        print("AA: ", end=', \t')
        for k in top_k_values:
            print(get_conf(percent_aa[k]), end='& ')
        print("")

        print("DCAA: ", end=', \t')
        for k in top_k_values:
            print(get_conf(percent_dcaa[k]), end='& ')
        print("")


    else:
        for k in top_k_values:
            after_imp.append(np.mean(percent_imp[k]) * 100)
            after_imp_err.append(np.std(percent_imp[k]) / np.sqrt(len(percent_imp[k])) * 100)

        print("After PYMK:")

        print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")

        print("CN: ", end=', \t')
        for k in top_k_values:
            print(get_conf(percent_cn[k]), end='& ')
        print("")

        print("DCCN: ", end=', \t')
        for k in top_k_values:
            print(get_conf(percent_dccn[k]), end='& ')
        print("")

        print("AA: ", end=', \t')
        for k in top_k_values:
            print(get_conf(percent_aa[k]), end='& ')
        print("")

        print("DCAA: ", end=', \t')
        for k in top_k_values:
            print(get_conf(percent_dcaa[k]), end='& ')
        print("")
#
#
# plt.figure()
# plt.rc('legend', fontsize=17)
# plt.rc('xtick', labelsize=12)
# plt.rc('ytick', labelsize=12)
# plt.errorbar(top_k_values, lower_imp, yerr=lower_imp_err, marker='o', color='b', ecolor='r', elinewidth=2,
#              label="Before PYMK")
# plt.errorbar(top_k_values, after_imp, yerr=after_imp_err, marker='^', color='g', ecolor='y', elinewidth=2,
#              label="After PYMK")
#
# plt.ylabel('AA vs CN Percent Improvement', fontsize=17)
# plt.xlabel('Top K Value', fontsize=17)
# plt.legend(loc='lower right')
# current_fig = plt.gcf()
# current_fig.savefig('{0}/plots/aa-cn.pdf'.format(path), format='pdf')
# plt.clf()



################ GOOGLE + ####################
# def calc_percent_imp(percent_imp_list, results, base_score, improved_score, ki):
#     if results[base_score][ki] != 0:
#         percent_imp_list[ki].append((results[improved_score][ki] - results[base_score][ki]) / results[base_score][ki])
#     else:
#         percent_imp_list[ki].append(results[improved_score][ki])
#
#
# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/'
# plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
# aa = {}
# dcaa = {}
# cn = {}
# dccn = {}
#
# # # Computing percent improvements and standard errors over all the ego nets analyzed
# percent_imp = {}
#
# for k in top_k_values:
#     percent_imp[k] = []
#     aa[k] = []
#     dcaa[k] = []
#     cn[k] = []
#     dccn[k] = []
#
# # loading result data
# for result_file in os.listdir(result_file_base_path + 'results'):
#     with open(result_file_base_path + 'results/' + result_file, 'rb') as f:
#         egonet_lp_results = pickle.load(f)
#
#     for k in top_k_values:
#         calc_percent_imp(percent_imp, egonet_lp_results, 'cn', 'aa', k)
#         aa[k].append(egonet_lp_results['aa'][k])
#         dcaa[k].append(egonet_lp_results['dcaa'][k])
#         cn[k].append(egonet_lp_results['cn'][k])
#         dccn[k].append(egonet_lp_results['dccn'][k])
#
# # imp_mse = {
# #     'imp': [],
# #     'imp_err': [],
# # }
# #
# # for k in top_k_values:
# #     imp_mse['imp'].append(np.mean(percent_imp[k]) * 100)
# #     imp_mse['imp_err'].append(np.std(percent_imp[k]) / np.sqrt(len(percent_imp[k])) * 100)
#
# # plt.figure()
# # plt.rc('legend', fontsize=17)
# # plt.rc('xtick', labelsize=12)
# # plt.rc('ytick', labelsize=12)
# #
# # plt.errorbar(top_k_values, imp_mse['imp'], yerr=imp_mse['imp_err'], marker='o', color='b', ecolor='r', elinewidth=2,
# #              label="AA VS CN")
# #
# # plt.ylabel("Percent Improvement", fontsize=15)
# # plt.xlabel('Top K Value', fontsize=15)
# # plt.legend(loc='lower right')
# #
# # current_fig = plt.gcf()
# # current_fig.savefig('{0}/gplus-lp-combined-aa-cn.pdf'.format(plot_save_path), format='pdf')
# # plt.clf()
# #
# # print("Plotting is Done!")
#
#
# print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")
#
# print("CN: ", end=', \t')
# for k in top_k_values:
#     print(get_conf(cn[k]), end='& ')
# print("")
#
# print("DCCN: ", end=', \t')
# for k in top_k_values:
#     print(get_conf(dccn[k]), end='& ')
# print("")
#
# print("AA: ", end=', \t')
# for k in top_k_values:
#     print(get_conf(aa[k]), end='& ')
# print("")
#
# print("DCAA: ", end=', \t')
# for k in top_k_values:
#     print(get_conf(dcaa[k]), end='& ')
# print("")