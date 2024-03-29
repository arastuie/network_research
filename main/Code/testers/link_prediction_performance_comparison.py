import os
import pickle
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def get_conf(list):
    m = np.mean(list) * 100
    err = 100 * np.std(list) / np.sqrt(len(list))

    # up = round(m + err, 2)
    # down = round(m - err, 2)
    # return "({0}, {1})".format(down, up)
    return "{0}^{1}".format(round(m, 2), round(err, 2))


def does_conf_overlap(base, update, k):
    bm = np.mean(base) * 100
    berr = 100 * np.std(base) / np.sqrt(len(base))

    um = np.mean(update) * 100
    uerr = 100 * np.std(update) / np.sqrt(len(update))

    if (bm + berr) < (um - uerr):
        print("@ K = {0}".format(k))
    elif (bm + berr) == (um - uerr):
        print("equal @ K = {0}".format(k))


# path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb'
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
# paths = ['lower-6', 'after-6']
# # paths = ['lower-6']
#
# lower_imp = []
# lower_imp_err = []
# after_imp = []
# after_imp_err = []
# for p in paths:
#     percent_imp = {}
#
#     for k in top_k_values:
#         percent_imp[k] = []
#
#     with open('{0}/pickle-files/{1}/temp-3/total-result.pckl'.format(path, p), 'rb') as f:
#         percent_aa, percent_dcaa, percent_cn, percent_dccn = pickle.load(f)
#
#     # with open('{0}/pickle-files/{1}/temp-4/total-result.pckl'.format(path, p), 'rb') as f:
#     #     percent_aa, percent_cn = pickle.load(f)
#
#         # for k in top_k_values:
#         #     for i in range(len(percent_cn[k])):
#         #         if percent_cn[k][i] != 0:
#         #             percent_imp[k].append((percent_aa[k][i] - percent_cn[k][i]) / percent_cn[k][i])
#         #         else:
#         #             percent_imp[k].append(percent_aa[k][i])
#
#     if p == 'lower-6':
#         for k in top_k_values:
#             lower_imp.append(np.mean(percent_imp[k]) * 100)
#             lower_imp_err.append(np.std(percent_imp[k]) / np.sqrt(len(percent_imp[k])) * 100)
#
#         print("Before PYMK:")
#
#         print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")
#
#         print("CAR: ", end=', \t')
#         for k in top_k_values:
#             print(get_conf(percent_cn[k]), end='& ')
#         print("")
#
#         print("DCCN: ", end=', \t')
#         for k in top_k_values:
#             print(get_conf(percent_dccn[k]), end='& ')
#         print("")
#
#         print("CCLP: ", end=', \t')
#         for k in top_k_values:
#             print(get_conf(percent_aa[k]), end='& ')
#         print("")
#
#         print("DCAA: ", end=', \t')
#         for k in top_k_values:
#             print(get_conf(percent_dcaa[k]), end='& ')
#         print("")
#
#         # print("\nCN vs. DCCN: ")
#         # for k in top_k_values:
#         #     does_conf_overlap(percent_cn[k], percent_dccn[k], k)
#         #
#         # print("\nAA vs. DCAA: ")
#         # for k in top_k_values:
#         #     does_conf_overlap(percent_aa[k], percent_dcaa[k], k)
#
#     else:
#         for k in top_k_values:
#             after_imp.append(np.mean(percent_imp[k]) * 100)
#             after_imp_err.append(np.std(percent_imp[k]) / np.sqrt(len(percent_imp[k])) * 100)
#
#         print("After PYMK:")
#
#         print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")
#
#         print("CAR: ", end=', \t')
#         for k in top_k_values:
#             print(get_conf(percent_cn[k]), end='& ')
#         print("")
#
#         print("DCCN: ", end=', \t')
#         for k in top_k_values:
#             print(get_conf(percent_dccn[k]), end='& ')
#         print("")
#
#         print("CCLP: ", end=', \t')
#         for k in top_k_values:
#             print(get_conf(percent_aa[k]), end='& ')
#         print("")
#
#         print("DCAA: ", end=', \t')
#         for k in top_k_values:
#             print(get_conf(percent_dcaa[k]), end='& ')
#         print("")

        # print("\nCN vs. DCCN: ")
        # for k in top_k_values:
        #     does_conf_overlap(percent_cn[k], percent_dccn[k], k)
        #
        # print("\nAA vs. DCAA: ")
        # for k in top_k_values:
        #     does_conf_overlap(percent_aa[k], percent_dcaa[k], k)

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



############### GOOGLE + ####################
# def calc_percent_imp(percent_imp_list, results, base_score, improved_score, ki):
#     if results[base_score][ki] != 0:
#         percent_imp_list[ki].append((results[improved_score][ki] - results[base_score][ki]) / results[base_score][ki])
#     else:
#         percent_imp_list[ki].append(results[improved_score][ki])
#
#
# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/test-2/'
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
# imp_mse = {
#     'imp': [],
#     'imp_err': [],
# }
#
# for k in top_k_values:
#     imp_mse['imp'].append(np.mean(percent_imp[k]) * 100)
#     imp_mse['imp_err'].append(np.std(percent_imp[k]) / np.sqrt(len(percent_imp[k])) * 100)
#
# plt.figure()
# plt.rc('legend', fontsize=17)
# plt.rc('xtick', labelsize=12)
# plt.rc('ytick', labelsize=12)
#
# plt.errorbar(top_k_values, imp_mse['imp'], yerr=imp_mse['imp_err'], marker='o', color='b', ecolor='r', elinewidth=2,
#              label="AA VS CN")
#
# plt.ylabel("Percent Improvement", fontsize=15)
# plt.xlabel('Top K Value', fontsize=15)
# plt.legend(loc='lower right')
#
# current_fig = plt.gcf()
# current_fig.savefig('{0}/gplus-lp-combined-aa-cn.pdf'.format(plot_save_path), format='pdf')
# plt.clf()
#
# print("Plotting is Done!")
#
# print(len(cn[k]))
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
#
# print("\nCN vs. DCCN: ")
# for k in top_k_values:
#     does_conf_overlap(cn[k], dccn[k], k)
#
# print("\nAA vs. DCAA: ")
# for k in top_k_values:
#     does_conf_overlap(aa[k], dcaa[k], k)

# ############### GOOGLE + w/ CAR and CCLP ####################
# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/'
# plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
# car = {}
# cclp = {}
# aa = {}
# dcaa = {}
# cn = {}
# dccn = {}
#
# both_result_directories = ['test-2/', 'test-3-car-cclp/']
#
# for k in top_k_values:
#     car[k] = []
#     cclp[k] = []
#     aa[k] = []
#     dcaa[k] = []
#     cn[k] = []
#     dccn[k] = []
#
# # loading result data
# for result_file in os.listdir(result_file_base_path + both_result_directories[0] + 'results'):
#
#     with open(result_file_base_path + both_result_directories[1] + 'results/' + result_file, 'rb') as f:
#         egonet_lp_results_car_cclp = pickle.load(f)
#
#     with open(result_file_base_path + both_result_directories[0] + 'results/' + result_file, 'rb') as f:
#         egonet_lp_results_first_four = pickle.load(f)
#
#     for k in top_k_values:
#         car[k].append(egonet_lp_results_car_cclp['car'][k])
#         cclp[k].append(egonet_lp_results_car_cclp['cclp'][k])
#         aa[k].append(egonet_lp_results_first_four['aa'][k])
#         dcaa[k].append(egonet_lp_results_first_four['dcaa'][k])
#         cn[k].append(egonet_lp_results_first_four['cn'][k])
#         dccn[k].append(egonet_lp_results_first_four['dccn'][k])
#
# # Write data into a single file
# with open(result_file_base_path + both_result_directories[0] + "all-6-methods-results.pckle", 'wb') as f:
#     pickle.dump([car, cclp, aa, dcaa, cn, dccn], f, protocol=-1)
#
# print("Number of egonets analyzed: {0}".format(len(cn[k])))
# print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")
#
# print("CAR& ", end=' \t')
# for k in top_k_values:
#     print(get_conf(car[k]), end='& ')
# print("")
#
# print("CCLP& ", end=' \t')
# for k in top_k_values:
#     print(get_conf(cclp[k]), end='& ')
# print("")
#
# print("CN& ", end=' \t')
# for k in top_k_values:
#     print(get_conf(cn[k]), end='& ')
# print("")
#
# print("DCCN& ", end=' \t')
# for k in top_k_values:
#     print(get_conf(dccn[k]), end='& ')
# print("")
#
# print("AA& ", end=' \t')
# for k in top_k_values:
#     print(get_conf(aa[k]), end='& ')
# print("")
#
# print("DCAA& ", end=' \t')
# for k in top_k_values:
#     print(get_conf(dcaa[k]), end='& ')
# print("")





############### Digg All LP Results ####################
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]

car = {}
cclp = {}
aa = {}
dcaa = {}
cn = {}
dccn = {}

both_result_directories = ['test-2/', 'test-3-car-cclp/']

for k in top_k_values:
    car[k] = []
    cclp[k] = []
    aa[k] = []
    dcaa[k] = []
    cn[k] = []
    dccn[k] = []

# loading result data
for result_file in os.listdir(result_file_base_path + both_result_directories[0] + 'results'):

    with open(result_file_base_path + both_result_directories[1] + 'results/' + result_file, 'rb') as f:
        egonet_lp_results_car_cclp = pickle.load(f)

    with open(result_file_base_path + both_result_directories[0] + 'results/' + result_file, 'rb') as f:
        egonet_lp_results_first_four = pickle.load(f)

    for k in top_k_values:
        car[k].append(egonet_lp_results_car_cclp['car'][k])
        cclp[k].append(egonet_lp_results_car_cclp['cclp'][k])
        aa[k].append(egonet_lp_results_first_four['aa'][k])
        dcaa[k].append(egonet_lp_results_first_four['dcaa'][k])
        cn[k].append(egonet_lp_results_first_four['cn'][k])
        dccn[k].append(egonet_lp_results_first_four['dccn'][k])

# Write data into a single file
with open(result_file_base_path + both_result_directories[0] + "all-6-methods-results.pckle", 'wb') as f:
    pickle.dump([car, cclp, aa, dcaa, cn, dccn], f, protocol=-1)

print("Number of egonets analyzed: {0}".format(len(cn[top_k_values[0]])))
print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")

print("CAR& ", end=' \t')
for k in top_k_values:
    print(get_conf(car[k]), end='& ')
print("")

print("CCLP& ", end=' \t')
for k in top_k_values:
    print(get_conf(cclp[k]), end='& ')
print("")

print("CN& ", end=' \t')
for k in top_k_values:
    print(get_conf(cn[k]), end='& ')
print("")

print("DCCN& ", end=' \t')
for k in top_k_values:
    print(get_conf(dccn[k]), end='& ')
print("")

print("AA& ", end=' \t')
for k in top_k_values:
    print(get_conf(aa[k]), end='& ')
print("")

print("DCAA& ", end=' \t')
for k in top_k_values:
    print(get_conf(dcaa[k]), end='& ')
print("")

