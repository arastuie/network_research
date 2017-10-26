import os
import pickle
import numpy as np
import helpers as h
import matplotlib.pyplot as plt

# Results pickle files are in the following order
#   local-formed-in-degree, global-formed-in-degree, local-formed-out-degree, global-formed-out-degree
#   local-not-formed-in-degree, global-not-formed-in-degree, local-not-formed-out-degree, global-not-formed-out-degree

triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/cdf-plots'
empirical_temp_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/temp'
gl_labels = ['Local', 'Global']
in_out_label = ['In-degree', 'Out-degree']
alpha = 0.05

# grabbing all scores
for triangle_type in triangle_types:
    results = []

    # loading result data
    for result_file in os.listdir(result_file_base_path + triangle_type):
        with open(result_file_base_path + triangle_type + '/' + result_file, 'rb') as f:
            egonet_result = pickle.load(f)

        results.append(egonet_result)
    results = np.array(results)

#     with open(result_file_base_path + 'all-scores/' + triangle_type + '.pckle', 'wb') as f:
#         pickle.dump(results, f, protocol=-1)
#
#     print(triangle_type + ": Done")
#
# print("Done!")
#
#
# # # plotting
# for triangle_type in triangle_types:
#     with open(result_file_base_path + 'all-scores/' + triangle_type + '.pckle', 'rb') as f:

    # for i in range(0, 2):
        # plt.hist(results[:, i], 100, normed=1, facecolor='g', alpha=0.75)
        # plt.suptitle('{0} - {1} Indegree Formed Edges'.format(triangle_type, gl_labels[i]))
        # plt.axvline(np.mean(results[:, i]), color='r', label='Mean')
        # plt.axvline(np.median(results[:, i]), color='b', label='Median')
        # plt.legend(loc='top right')
        # plt.savefig('{0}/{1}-{2}-cdf.png'.format(empirical_temp_path, triangle_type, i))
        # plt.clf()
        #
        # plt.hist(results[:, i + 4], 100, normed=1, facecolor='g', alpha=0.75)
        # plt.suptitle('{0} - {1} Indegree Not Formed Edges'.format(triangle_type, gl_labels[i]))
        # plt.axvline(np.mean(results[:, i + 4]), color='r', label='Mean')
        # plt.axvline(np.median(results[:, i + 4]), color='b', label='Median')
        # plt.legend(loc='top right')
        # plt.savefig('{0}/{1}-{2}-cdf.png'.format(empirical_temp_path, triangle_type, i + 4))
        # plt.clf()
        #
        # plt.hist(results[:, i + 2], 100, normed=1, facecolor='g', alpha=0.75)
        # plt.suptitle('{0} - {1} Outdegree Formed Edges'.format(triangle_type, gl_labels[i]))
        # plt.axvline(np.mean(results[:, i + 2]), color='r', label='Mean')
        # plt.axvline(np.median(results[:, i + 2]), color='b', label='Median')
        # plt.legend(loc='top right')
        # plt.savefig('{0}/{1}-{2}-cdf.png'.format(empirical_temp_path, triangle_type, i + 2))
        # plt.clf()
        #
        # plt.hist(results[:, i + 6], 100, normed=1, facecolor='g', alpha=0.75)
        # plt.suptitle('{0} - {1} Outdegree Not Formed Edges'.format(triangle_type, gl_labels[i]))
        # plt.axvline(np.mean(results[:, i + 6]), color='r', label='Mean')
        # plt.axvline(np.median(results[:, i + 6]), color='b', label='Median')
        # plt.legend(loc='top right')
        # plt.savefig('{0}/{1}-{2}-cdf.png'.format(empirical_temp_path, triangle_type, i + 6))
        # plt.clf()


        # plt.rc('legend', fontsize=12)
        # plt.rc('xtick', labelsize=12)
        # plt.rc('ytick', labelsize=12)
        #
        # h.add_ecdf_with_bond_plot(res[:, i], lbs[i], ubs[i], 'Indegree Formed Edges', 'r')
        # h.add_ecdf_with_bond_plot(res[:, i + 4], lbs[i + 4], ubs[i + 4], 'Indegree Not Formed Edges', 'b')
        #
        # h.add_ecdf_with_bond_plot(res[:, i + 2], lbs[i + 2], ubs[i + 2], 'Outdegree Formed Edges', 'y')
        # h.add_ecdf_with_bond_plot(res[:, i + 6], lbs[i + 6], ubs[i + 6], 'Outdegree Not Formed Edges', 'g')
        #
        # plt.ylabel('Empirical CDF', fontsize=14)
        # plt.xlabel('Mean Normalized {0} Degree'.format(gl_labels[i]), fontsize=14)
        # plt.legend(loc='lower right')
        # plt.suptitle('Number of {0} egonets analyzed: {1}'.format(triangle_type, len(res)))
        # current_fig = plt.gcf()
        # current_fig.savefig('{0}/{1}-{2}-cdf.pdf'.format(plot_save_path, triangle_type, gl_labels[i]), format='pdf')
        # # current_fig.savefig('{0}/{1}-{2}-cdf.png'.format(plot_save_path, triangle_type, gl_labels[i]))
        # plt.clf()

    bar_color = ['r', 'b', 'g', 'y']
    for i in range(2):
        # divided into in- & out- degree. With the following order:
        # global-formed, global-not-formed, local-formed, local-not-formed
        rep_values = [np.mean(results[:, i * 2 + 1]), np.mean(results[:, i * 2 + 5]), np.mean(results[:, i * 2]), np.mean(results[:, i * 2 + 4])]
        plt.figure(i + 1)
        plt.bar()

    print(triangle_type + ": Done")

print("Done!")
