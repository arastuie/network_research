import os
import math
import pickle
import numpy as np
import Code.helpers as h
import matplotlib.pyplot as plt

# Results pickle files are in the following order
#   local-formed-in-degree, global-formed-in-degree, local-formed-out-degree, global-formed-out-degree
#   local-not-formed-in-degree, global-not-formed-in-degree, local-not-formed-out-degree, global-not-formed-out-degree

triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files/' \
                        'test-2-no-least-num-nodes/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/cdf-plots'

gl_labels = ['Local', 'Global']
# alpha = 0.05
# # grabbing all scores
# for triangle_type in triangle_types:
#     results = []
#
#     # loading result data
#     for result_file in os.listdir(result_file_base_path + triangle_type):
#         with open(result_file_base_path + triangle_type + '/' + result_file, 'rb') as f:
#             egonet_result = pickle.load(f)
#
#         results.append(egonet_result)
#     results = np.array(results)
#
#     lower_bonds = []
#     upper_bonds = []
#     for ii in range(8):
#         lb, ub = h.get_ecdf_bands(results[:, ii], alpha)
#         lower_bonds.append(lb)
#         upper_bonds.append(ub)
#
#     with open(result_file_base_path + 'all-scores/' + triangle_type + '.pckle', 'wb') as f:
#         pickle.dump([results, lower_bonds, upper_bonds], f, protocol=-1)
#
#     print(triangle_type + ": Done")
#
# print("Done!")


# plotting
# for triangle_type in triangle_types:
#     with open(result_file_base_path + 'all-scores/' + triangle_type + '.pckle', 'rb') as f:
#         res, lbs, ubs = pickle.load(f)
#
#     for i in range(0, 2):
#         plt.rc('legend', fontsize=14.5)
#         plt.rc('xtick', labelsize=15)
#         plt.rc('ytick', labelsize=15)
#
#         h.add_ecdf_with_band_plot(res[:, i], lbs[i], ubs[i], 'In-degree Formed', 'r')
#         h.add_ecdf_with_band_plot(res[:, i + 4], lbs[i + 4], ubs[i + 4], 'In-degree Not Formed', 'b')
#
#         h.add_ecdf_with_band_plot(res[:, i + 2], lbs[i + 2], ubs[i + 2], 'Out-degree Formed', 'y')
#         h.add_ecdf_with_band_plot(res[:, i + 6], lbs[i + 6], ubs[i + 6], 'Out-degree Not Formed', 'g')
#
#         plt.ylabel('Empirical CDF', fontsize=20)
#         plt.xlabel('Mean Normalized {0} Degree'.format(gl_labels[i]), fontsize=20)
#         plt.legend(loc='lower right')
#         # plt.suptitle('Number of {0} egonets analyzed: {1}'.format(triangle_type, len(res)))
#         plt.tight_layout()
#         current_fig = plt.gcf()
#         current_fig.savefig('{0}/{1}-{2}-cdf.pdf'.format(plot_save_path, triangle_type, gl_labels[i]), format='pdf')
#         # current_fig.savefig('{0}/{1}-{2}-cdf.png'.format(plot_save_path, triangle_type, gl_labels[i]))
#         plt.clf()
#
#     print(triangle_type + ": Done")
#
# print("Done!")
#
# for triangle_type in ['T02']:
#     with open(result_file_base_path + 'all-scores/' + triangle_type + '.pckle', 'rb') as f:
#         res, lbs, ubs = pickle.load(f)
#
#     for i in range(0, 2):
#         plt.rc('legend', fontsize=20)
#         plt.rc('xtick', labelsize=15)
#         plt.rc('ytick', labelsize=15)
#
#         h.add_ecdf_with_band_plot(res[:, i], lbs[i], ubs[i], 'Formed', 'r')
#         h.add_ecdf_with_band_plot(res[:, i + 4], lbs[i + 4], ubs[i + 4], 'Not Formed', 'b')
#
#         plt.ylabel('Empirical CDF', fontsize=20)
#         plt.xlabel('Mean Normalized {0} In-degree'.format(gl_labels[i]), fontsize=20)
#         plt.legend(loc='lower right')
#         plt.tight_layout()
#         current_fig = plt.gcf()
#         current_fig.savefig('{0}/{1}-indegree-{2}-cdf-test-2.pdf'.format(plot_save_path, triangle_type, gl_labels[i]), format='pdf')
#         # current_fig.savefig('{0}/{1}-{2}-cdf.png'.format(plot_save_path, triangle_type, gl_labels[i]))
#         plt.clf()
#
#         plt.rc('legend', fontsize=20)
#         plt.rc('xtick', labelsize=15)
#         plt.rc('ytick', labelsize=15)
#
#         h.add_ecdf_with_band_plot(res[:, i + 2], lbs[i + 2], ubs[i + 2], 'Formed', 'r')
#         h.add_ecdf_with_band_plot(res[:, i + 6], lbs[i + 6], ubs[i + 6], 'Not Formed', 'b')
#
#         plt.ylabel('Empirical CDF', fontsize=20)
#         plt.xlabel('Mean Normalized {0} Out-degree'.format(gl_labels[i]), fontsize=20)
#         plt.legend(loc='lower right')
#         plt.tight_layout()
#         current_fig = plt.gcf()
#         current_fig.savefig('{0}/{1}-outdegree-{2}-cdf-test-2.pdf'.format(plot_save_path, triangle_type, gl_labels[i]), format='pdf')
#         # current_fig.savefig('{0}/{1}-{2}-cdf.png'.format(plot_save_path, triangle_type, gl_labels[i]))
#         plt.clf()
#
#     print(triangle_type + ": Done")
#
# print("Done!")
