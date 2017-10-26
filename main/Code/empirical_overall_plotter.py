import os
import pickle
import numpy as np
import helpers as h
import matplotlib.pyplot as plt


def get_mean_ci(res, z_value):
    return z_value * np.std(res) / np.sqrt(len(res))

# Results pickle files are in the following order
#   local-formed-in-degree, global-formed-in-degree, local-formed-out-degree, global-formed-out-degree
#   local-not-formed-in-degree, global-not-formed-in-degree, local-not-formed-out-degree, global-not-formed-out-degree

triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/cdf-plots'

gl_labels = ['Local', 'Global']
in_out_label = ['In-degree', 'Out-degree']
z = 1.96


pos = list(range(len(triangle_types)))
bar_width = 0.20
opacity = 0.6
error_config = {'ecolor': '0.3'}
bar_legends = ['Global Formed', 'Global Not Formed', 'Local Formed', 'Local Not Formed']
dif_results = ['global-formed', 'global-not-formed', 'local-formed', 'local-not-formed']
bar_color = ['r', 'b', 'g', 'y']


all_results = {
    'In-degree': {
        'global-formed': [],
        'global-not-formed': [],
        'local-formed': [],
        'local-not-formed': [],
        'global-formed-err': [],
        'global-not-formed-err': [],
        'local-formed-err': [],
        'local-not-formed-err': []
    },
    'Out-degree': {
        'global-formed': [],
        'global-not-formed': [],
        'local-formed': [],
        'local-not-formed': [],
        'global-formed-err': [],
        'global-not-formed-err': [],
        'local-formed-err': [],
        'local-not-formed-err': []
    }
}

# grabbing all scores
for t in range(len(triangle_types)):
    results = []

    # loading result data
    for result_file in os.listdir(result_file_base_path + triangle_types[t]):
        with open(result_file_base_path + triangle_types[t] + '/' + result_file, 'rb') as f:
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

    for i in range(2):
        # computing mean
        all_results[in_out_label[i]]['global-formed'].append(np.mean(results[:, i * 2 + 1]))
        all_results[in_out_label[i]]['global-not-formed'].append(np.mean(results[:, i * 2 + 5]))
        all_results[in_out_label[i]]['local-formed'].append(np.mean(results[:, i * 2]))
        all_results[in_out_label[i]]['local-not-formed'].append(np.mean(results[:, i * 2 + 4]))

        # computing 95% confidence interval
        all_results[in_out_label[i]]['global-formed-err'].append(get_mean_ci(results[:, i * 2 + 1], z))
        all_results[in_out_label[i]]['global-not-formed-err'].append(get_mean_ci(results[:, i * 2 + 5], z))
        all_results[in_out_label[i]]['local-formed-err'].append(get_mean_ci(results[:, i * 2], z))
        all_results[in_out_label[i]]['local-not-formed-err'].append(get_mean_ci(results[:, i * 2 + 4], z))

    print(triangle_types[t] + ": Done")

# plotting
for i_degree in range(2):
    fig, ax = plt.subplots()

    for i_bar in range(len(dif_results)):
        plt.bar(np.arange(len(triangle_types)) + bar_width * i_bar,
                all_results[in_out_label[i_degree]][dif_results[i_bar]],
                bar_width,
                alpha=opacity,
                color=bar_color[i_bar],
                yerr=all_results[in_out_label[i_degree]][dif_results[i_bar] + '-err'],
                error_kw=error_config,
                label=bar_legends[i_bar])

    plt.xlabel('Triangle Types')
    plt.ylabel('Mean Normalized {0} of Common Neighbors'.format(in_out_label[i_degree]))
    plt.xticks(np.arange(len(triangle_types)) + bar_width * 1.5, triangle_types)

    if i_degree == 1:
        plt.ylim(ymax=0.4)
        plt.legend(loc='upper center')
    else:
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.grid()
    plt.savefig('{0}/overall-{1}.pdf'.format(plot_save_path, in_out_label[i_degree]), format='pdf')
    plt.clf()
