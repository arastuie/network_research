import pickle
import numpy as np
import matplotlib.pyplot as plt

path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]

paths = ['lower-6', 'after-6']

# for p in paths:
#     percent_imp_aa = {}
#     percent_imp_cn = {}
#
#     for k in top_k_values:
#         percent_imp_aa[k] = []
#         percent_imp_cn[k] = []
#
#     with open('{0}/{1}/temp/total-result.pckl'.format(path, p), 'rb') as f:
#         percent_aa, percent_dcaa, percent_cn, percent_dccn = pickle.load(f)
#
#         print(len(percent_aa[1]))
#         print(len(percent_dcaa[1]))
#         print(len(percent_cn[1]))
#         print(len(percent_dccn[1]))
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
#     with open('{0}/{1}/temp/total-result-percent-imp.pckl'.format(path, p), 'wb') as f:
#         pickle.dump([percent_imp_aa, percent_imp_cn], f, protocol=-1)

with open('{0}/pickle-files/lower-6/temp/total-result-percent-imp.pckl'.format(path), 'rb') as f:
    lower_percent_imp_aa, lower_percent_imp_cn = pickle.load(f)
lower_imp_aa = []
lower_imp_aa_err = []
lower_imp_cn = []
lower_imp_cn_err = []
for k in top_k_values:
    lower_imp_aa.append(np.mean(lower_percent_imp_aa[k]) * 100)
    lower_imp_aa_err.append(np.std(lower_percent_imp_aa[k]) / np.sqrt(len(lower_percent_imp_aa[k])) * 100)
    lower_imp_cn.append(np.mean(lower_percent_imp_cn[k]) * 100)
    lower_imp_cn_err.append(np.std(lower_percent_imp_cn[k]) / np.sqrt(len(lower_percent_imp_cn[k])) * 100)


with open('{0}/pickle-files/after-6/temp/total-result-percent-imp.pckl'.format(path), 'rb') as f:
    after_percent_imp_aa, after_percent_imp_cn = pickle.load(f)
after_imp_aa = []
after_imp_aa_err = []
after_imp_cn = []
after_imp_cn_err = []
for k in top_k_values:
    after_imp_aa.append(np.mean(after_percent_imp_aa[k]) * 100)
    after_imp_aa_err.append(np.std(after_percent_imp_aa[k]) / np.sqrt(len(after_percent_imp_aa[k])) * 100)
    after_imp_cn.append(np.mean(after_percent_imp_cn[k]) * 100)
    after_imp_cn_err.append(np.std(after_percent_imp_cn[k]) / np.sqrt(len(after_percent_imp_cn[k])) * 100)

print("lower imp aa -> {0}".format(np.mean(lower_imp_aa)))
print("after imp aa -> {0}".format(np.mean(after_imp_aa)))
print("lower imp cn -> {0}".format(np.mean(lower_imp_cn)))
print("after imp cn -> {0}".format(np.mean(after_imp_cn)))
print("overall -> {0}".format(np.mean([np.mean(lower_imp_aa), np.mean(after_imp_aa), np.mean(lower_imp_cn), np.mean(after_imp_cn) ])))

plt.figure()
plt.rc('legend', fontsize=17)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.errorbar(top_k_values, lower_imp_aa, yerr=lower_imp_aa_err, marker='o', ecolor='b', elinewidth=3,
             label="Before PYMK")
plt.errorbar(top_k_values, after_imp_aa, yerr=after_imp_aa_err, marker='^', ecolor='r', elinewidth=3,
             label="After PYMK")

plt.ylabel('DCAA vs AA Percent Improvement', fontsize=17)
plt.xlabel('Top K Value', fontsize=17)
plt.legend(loc='lower right')
current_fig = plt.gcf()
current_fig.savefig('{0}/plots/dcaa-aa.pdf'.format(path), format='pdf')
plt.clf()

plt.figure()
plt.rc('legend', fontsize=17)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.errorbar(top_k_values, lower_imp_cn, yerr=lower_imp_cn_err, marker='o', ecolor='b', elinewidth=3,
             label="Before PYMK")
plt.errorbar(top_k_values, after_imp_cn, yerr=after_imp_cn_err, marker='^', ecolor='r', elinewidth=3,
             label="After PYMK")

plt.ylabel('DCCN vs CN Percent Improvement', fontsize=17)
plt.xlabel('Top K Value', fontsize=17)
plt.legend(loc='lower right')


current_fig = plt.gcf()
current_fig.savefig('{0}/plots/dccn-cn.pdf'.format(path), format='pdf')
plt.clf()




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