import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb'
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
# paths = ['lower-6', 'after-6']

# for p in paths:
#     percent_imp_aa = {}
#     percent_imp_cn = {}
#
#     for k in top_k_values:
#         percent_imp_aa[k] = []
#         percent_imp_cn[k] = []
#
#     with open('{0}/pickle-files/{1}/temp-2/total-result.pckl'.format(path, p), 'rb') as f:
#         percent_aa, percent_dcaa, percent_cn, percent_dccn = pickle.load(f)
#
#         for k in top_k_values:
#             for i in range(len(percent_aa[k])):
#                 if percent_aa[k][i] != 0:
#                     percent_imp_aa[k].append((percent_dcaa[k][i] - percent_aa[k][i]) / percent_aa[k][i])
#                 else:
#                     percent_imp_aa[k].append(percent_dcaa[k][i])
#
#                 if percent_cn[k][i] != 0:
#                     percent_imp_cn[k].append((percent_dccn[k][i] - percent_cn[k][i]) / percent_cn[k][i])
#                 else:
#                     percent_imp_cn[k].append(percent_dccn[k][i])
#
#     with open('{0}/pickle-files/{1}/temp-2/total-result-percent-imp.pckl'.format(path, p), 'wb') as f:
#         pickle.dump([percent_imp_aa, percent_imp_cn], f, protocol=-1)
#
#
# with open('{0}/pickle-files/lower-6/temp-2/total-result-percent-imp.pckl'.format(path), 'rb') as f:
#     lower_percent_imp_aa, lower_percent_imp_cn = pickle.load(f)
# lower_imp_aa = []
# lower_imp_aa_err = []
# lower_imp_cn = []
# lower_imp_cn_err = []
# for k in top_k_values:
#     lower_imp_aa.append(np.mean(lower_percent_imp_aa[k]) * 100)
#     lower_imp_aa_err.append(np.std(lower_percent_imp_aa[k]) / np.sqrt(len(lower_percent_imp_aa[k])) * 100)
#     lower_imp_cn.append(np.mean(lower_percent_imp_cn[k]) * 100)
#     lower_imp_cn_err.append(np.std(lower_percent_imp_cn[k]) / np.sqrt(len(lower_percent_imp_cn[k])) * 100)
#
#
# with open('{0}/pickle-files/after-6/temp-2/total-result-percent-imp.pckl'.format(path), 'rb') as f:
#     after_percent_imp_aa, after_percent_imp_cn = pickle.load(f)
# after_imp_aa = []
# after_imp_aa_err = []
# after_imp_cn = []
# after_imp_cn_err = []
# for k in top_k_values:
#     after_imp_aa.append(np.mean(after_percent_imp_aa[k]) * 100)
#     after_imp_aa_err.append(np.std(after_percent_imp_aa[k]) / np.sqrt(len(after_percent_imp_aa[k])) * 100)
#     after_imp_cn.append(np.mean(after_percent_imp_cn[k]) * 100)
#     after_imp_cn_err.append(np.std(after_percent_imp_cn[k]) / np.sqrt(len(after_percent_imp_cn[k])) * 100)
#
# plt.figure()
# plt.rc('legend', fontsize=25)
# plt.rc('xtick', labelsize=17)
# plt.rc('ytick', labelsize=17)
# plt.errorbar(top_k_values, lower_imp_aa, yerr=lower_imp_aa_err, marker='o', color='b', ecolor='r', elinewidth=2,
#              label="Before PYMK")
# plt.errorbar(top_k_values, after_imp_aa, yerr=after_imp_aa_err, marker='^', color='g', ecolor='y', elinewidth=2,
#              label="After PYMK")
#
# plt.ylabel('Percent Improvement', fontsize=22)
# plt.xlabel('Top K Value', fontsize=22)
# plt.legend(loc='lower right')
# plt.tight_layout()
# current_fig = plt.gcf()
# current_fig.savefig('{0}/plots/dcaa-aa.pdf'.format(path), format='pdf')
# plt.clf()
#
# plt.figure()
# plt.rc('legend', fontsize=25)
# plt.rc('xtick', labelsize=17)
# plt.rc('ytick', labelsize=17)
# plt.errorbar(top_k_values, lower_imp_cn, yerr=lower_imp_cn_err, marker='o', color='b', ecolor='r', elinewidth=2,
#              label="Before PYMK")
# plt.errorbar(top_k_values, after_imp_cn, yerr=after_imp_cn_err, marker='^', color='g', ecolor='y', elinewidth=2,
#              label="After PYMK")
#
# plt.ylabel('Percent Improvement', fontsize=22)
# plt.xlabel('Top K Value', fontsize=22)
# plt.legend(loc='lower right')
# plt.tight_layout()
# current_fig = plt.gcf()
# current_fig.savefig('{0}/plots/dccn-cn.pdf'.format(path), format='pdf')
# plt.clf()
#
# print("Done!")




# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
# path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files'
# top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
#
# paths = ['lower-6', 'after-6']
#
# for p in paths:
#     # percent_imp_aa = {}
#     # percent_imp_cn = {}
#     #
#     # for k in top_k_values:
#     #     percent_imp_aa[k] = []
#     #     percent_imp_cn[k] = []
#     #
#     # with open('{0}/{1}/temp/total-result.pckl'.format(path, p), 'rb') as f:
#     #     percent_aa, percent_dcaa, percent_cn, percent_dccn = pickle.load(f)
#     #
#     #     print(len(percent_aa[1]))
#     #     print(len(percent_dcaa[1]))
#     #     print(len(percent_cn[1]))
#     #     print(len(percent_dccn[1]))
#     #
#     #     for k in top_k_values:
#     #         for i in range(len(percent_aa[k])):
#     #             if percent_aa[k][i] != 0:
#     #                 percent_imp_aa[k].append((percent_dcaa[k][i] - percent_aa[k][i]) / percent_aa[k][i])
#     #             else:
#     #                 percent_imp_aa[k].append(percent_dcaa[k][i])
#     #
#     #             if percent_cn[k][i] != 0:
#     #                 percent_imp_cn[k].append((percent_dccn[k][i] - percent_cn[k][i]) / percent_cn[k][i])
#     #             else:
#     #                 percent_imp_cn[k].append(percent_dccn[k][i])
#     #
#     # with open('{0}/{1}/temp/total-result-percent-imp.pckl'.format(path, p), 'wb') as f:
#     #     pickle.dump([percent_imp_aa, percent_imp_cn], f, protocol=-1)
#
#     with open('{0}/{1}/temp/total-result-percent-imp.pckl'.format(path, p), 'rb') as f:
#         percent_imp_aa, percent_imp_cn = pickle.load(f)
#
#     imp_aa = []
#     imp_aa_err = []
#
#     imp_cn = []
#     imp_cn_err = []
#     for k in top_k_values:
#         imp_aa.append(np.mean(percent_imp_aa[k]))
#         imp_aa_err.append(np.std(percent_imp_aa[k]) / np.sqrt(len(percent_imp_aa[k])))
#
#         imp_cn.append(np.mean(percent_imp_cn[k]))
#         imp_cn_err.append(np.std(percent_imp_cn[k]) / np.sqrt(len(percent_imp_cn[k])))
#
#     plt.figure()
#     plt.errorbar(top_k_values, imp_aa, yerr=imp_aa_err, marker='o', ecolor='b', elinewidth=3)
#     if p == 'lower-6':
#         plt.title("Before PYMK", fontsize=14)
#     else:
#         plt.title("After PYMK", fontsize=14)
#
#     plt.ylabel('Top K Value', fontsize=13)
#     plt.xlabel('DCAA vs AA Percent Improvement', fontsize=13)
#
#     current_fig = plt.gcf()
#     current_fig.savefig('{0}/plots/{1}-dcaa-aa.eps'.format(path, p), format='eps')
#     plt.clf()
#
#     plt.figure()
#     plt.errorbar(top_k_values, imp_cn, yerr=imp_cn_err, marker='o', ecolor='b', elinewidth=3)
#     if p == 'lower-6':
#         plt.title("Before PYMK", fontsize=14)
#     else:
#         plt.title("After PYMK", fontsize=14)
#
#     plt.ylabel('Top K Value', fontsize=13)
#     plt.xlabel('DCCN vs CN Percent Improvement', fontsize=13)
#
#     current_fig = plt.gcf()
#     current_fig.savefig('{0}/plots/{1}-dccn-cn.eps'.format(path, p), format='eps')
#     plt.clf()




def eval_percent_imp(list, base_score, imp_score, ki):
    base_mean = np.mean(base_score[ki])
    if base_mean != 0:
        list[ki] = (np.mean(imp_score[ki]) - base_mean) / base_mean
    else:
        list[ki] = np.mean(imp_score[ki])


def eval_2_std(base_score, imp_score):
    n = len(imp_score)
    y_var = np.var(imp_score) / n
    y_mean = np.mean(imp_score)

    x_var = np.var(base_score) / n
    x_mean = np.mean(base_score)

    se = 200 * np.sqrt((y_var * (x_mean ** 2) + x_var * (y_mean ** 2)) / (x_mean ** 4))
    print(se)
    return se


def plot_it(top_K, res, err, path):
    ax = plt.figure().gca()
    plt.errorbar(top_K, res, marker='o', color='b', ecolor='r', elinewidth=2)
    plt.ylabel('Percent Improvement', fontsize=22)
    plt.xlabel('Top K Value', fontsize=22)
    plt.tight_layout()
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.yticks(np.arange(0, max(res) + 0.5, 0.5))
    current_fig = plt.gcf()
    current_fig.savefig(path, format='pdf')
    plt.clf()


path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]

paths = ['lower-6', 'after-6']

both_res = {
    'lower-6': {
        'imp_cn': [],
        'imp_cn_err': [],
        'imp_aa': [],
        'imp_aa_err': []
    },

    'after-6': {
        'imp_cn': [],
        'imp_cn_err': [],
        'imp_aa': [],
        'imp_aa_err': []
    }
}

for p in paths:
    percent_imp_aa = {}
    percent_imp_cn = {}

    for k in top_k_values:
        percent_imp_aa[k] = []
        percent_imp_cn[k] = []

    with open('{0}/pickle-files/{1}/temp-3/total-result.pckl'.format(path, p), 'rb') as f:
        percent_aa, percent_dcaa, percent_cn, percent_dccn = pickle.load(f)

    for k in top_k_values:
        eval_percent_imp(percent_imp_cn, percent_cn, percent_dccn, k)
        eval_percent_imp(percent_imp_aa, percent_aa, percent_dcaa, k)

    for k in top_k_values:
        both_res[p]['imp_cn'].append(percent_imp_cn[k] * 100)
        # both_res[p]['imp_cn_err'].append(eval_2_std(percent_cn[k], percent_dccn[k]))

        both_res[p]['imp_aa'].append(percent_imp_aa[k] * 100)
        # both_res[p]['imp_aa_err'].append(eval_2_std(percent_aa[k], percent_dcaa[k]))

    print(len(percent_aa[1]))


plot_it(top_k_values, both_res['after-6']['imp_aa'], both_res['after-6']['imp_aa_err'],
        '{0}/plots/dcaa-aa-after-3.pdf'.format(path))

plot_it(top_k_values, both_res['lower-6']['imp_aa'], both_res['lower-6']['imp_aa_err'],
        '{0}/plots/dcaa-aa-before-3.pdf'.format(path))

plot_it(top_k_values, both_res['lower-6']['imp_cn'], both_res['lower-6']['imp_cn_err'],
        '{0}/plots/dccn-cn-before-3.pdf'.format(path))

plot_it(top_k_values, both_res['after-6']['imp_cn'], both_res['after-6']['imp_cn_err'],
        '{0}/plots/dccn-cn-after-3.pdf'.format(path))

# plot_it(top_k_values, both_res['lower-6']['imp_aa'], both_res['lower-6']['imp_aa_err'],
#         '{0}/plots/ldaa-cclp-before-4.pdf'.format(path))
#
# plot_it(top_k_values, both_res['after-6']['imp_aa'], both_res['after-6']['imp_aa_err'],
#         '{0}/plots/ldaa-cclp-after-4.pdf'.format(path))
#
# plot_it(top_k_values, both_res['lower-6']['imp_cn'], both_res['lower-6']['imp_cn_err'],
#         '{0}/plots/ldcn-car-before-4.pdf'.format(path))
#
# plot_it(top_k_values, both_res['after-6']['imp_cn'], both_res['after-6']['imp_cn_err'],
#         '{0}/plots/ldcn-car-after-4.pdf'.format(path))

# plt.figure()
# plt.rc('legend', fontsize=25)
# plt.rc('xtick', labelsize=17)
# plt.rc('ytick', labelsize=17)
# plt.errorbar(top_k_values, both_res['lower-6']['imp_aa'], yerr=both_res['lower-6']['imp_aa_err'], marker='o', color='b',
#              ecolor='r', elinewidth=2, label="Before PYMK")
# plt.errorbar(top_k_values, both_res['after-6']['imp_aa'], yerr=both_res['after-6']['imp_aa_err'], marker='^', color='g',
#              ecolor='y', elinewidth=2, label="After PYMK")
#
# plt.ylabel('Percent Improvement', fontsize=22)
# plt.xlabel('Top K Value', fontsize=22)
# plt.legend(loc='lower right')
# plt.tight_layout()
# current_fig = plt.gcf()
# current_fig.savefig('{0}/plots/dcaa-aa-2.pdf'.format(path), format='pdf')
# plt.clf()
#
# plt.figure()
# plt.rc('legend', fontsize=25)
# plt.rc('xtick', labelsize=17)
# plt.rc('ytick', labelsize=17)
# plt.errorbar(top_k_values, both_res['lower-6']['imp_cn'], yerr=both_res['lower-6']['imp_cn_err'], marker='o', color='b',
#              ecolor='r', elinewidth=2, label="Before PYMK")
# plt.errorbar(top_k_values, both_res['after-6']['imp_cn'], yerr=both_res['after-6']['imp_cn_err'], marker='^', color='g',
#              ecolor='y', elinewidth=2, label="After PYMK")
#
# plt.ylabel('Percent Improvement', fontsize=22)
# plt.xlabel('Top K Value', fontsize=22)
# plt.legend(loc='lower right')
# plt.tight_layout()
# current_fig = plt.gcf()
# current_fig.savefig('{0}/plots/dccn-cn-2.pdf'.format(path), format='pdf')
# plt.clf()

print("Done!")