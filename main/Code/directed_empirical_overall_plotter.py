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


# triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files/'
# plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/cdf-plots'
#
# gl_labels = ['Local', 'Global']
# in_out_label = ['In-degree', 'Out-degree']
# z = 1.96
#
#
# pos = list(range(len(triangle_types)))
# bar_width = 0.20
# opacity = 0.6
# error_config = {'ecolor': '0.3'}
# bar_legends = ['Global Formed', 'Global Not Formed', 'Local Formed', 'Local Not Formed']
# dif_results = ['global-formed', 'global-not-formed', 'local-formed', 'local-not-formed']
# bar_color = ['r', 'b', 'g', 'y']
#
#
# all_results = {
#     'In-degree': {
#         'global-formed': [],
#         'global-not-formed': [],
#         'local-formed': [],
#         'local-not-formed': [],
#         'global-formed-err': [],
#         'global-not-formed-err': [],
#         'local-formed-err': [],
#         'local-not-formed-err': []
#     },
#     'Out-degree': {
#         'global-formed': [],
#         'global-not-formed': [],
#         'local-formed': [],
#         'local-not-formed': [],
#         'global-formed-err': [],
#         'global-not-formed-err': [],
#         'local-formed-err': [],
#         'local-not-formed-err': []
#     }
# }
#
# # grabbing all scores
# for t in range(len(triangle_types)):
#     results = []
#
#     # loading result data
#     for result_file in os.listdir(result_file_base_path + triangle_types[t]):
#         with open(result_file_base_path + triangle_types[t] + '/' + result_file, 'rb') as f:
#             egonet_result = pickle.load(f)
#
#         results.append(egonet_result)
#     results = np.array(results)
#
#     for i in range(2):
#         # computing mean
#         all_results[in_out_label[i]]['global-formed'].append(np.mean(results[:, i * 2 + 1]))
#         all_results[in_out_label[i]]['global-not-formed'].append(np.mean(results[:, i * 2 + 5]))
#         all_results[in_out_label[i]]['local-formed'].append(np.mean(results[:, i * 2]))
#         all_results[in_out_label[i]]['local-not-formed'].append(np.mean(results[:, i * 2 + 4]))
#
#         # computing 95% confidence interval
#         all_results[in_out_label[i]]['global-formed-err'].append(get_mean_ci(results[:, i * 2 + 1], z))
#         all_results[in_out_label[i]]['global-not-formed-err'].append(get_mean_ci(results[:, i * 2 + 5], z))
#         all_results[in_out_label[i]]['local-formed-err'].append(get_mean_ci(results[:, i * 2], z))
#         all_results[in_out_label[i]]['local-not-formed-err'].append(get_mean_ci(results[:, i * 2 + 4], z))
#
#     print(triangle_types[t] + ": Done")
#
# # plotting
# for i_degree in range(2):
#     fig, ax = plt.subplots()
#
#     for i_bar in range(len(dif_results)):
#         plt.bar(np.arange(len(triangle_types)) + bar_width * i_bar,
#                 all_results[in_out_label[i_degree]][dif_results[i_bar]],
#                 bar_width,
#                 alpha=opacity,
#                 color=bar_color[i_bar],
#                 yerr=all_results[in_out_label[i_degree]][dif_results[i_bar] + '-err'],
#                 error_kw=error_config,
#                 label=bar_legends[i_bar])
#
#     plt.xlabel('Triangle Types')
#     plt.ylabel('Mean Normalized {0} of Common Neighbors'.format(in_out_label[i_degree]))
#     plt.xticks(np.arange(len(triangle_types)) + bar_width * 1.5, triangle_types)
#
#     if i_degree == 1:
#         plt.ylim(ymax=0.4)
#         plt.legend(loc='upper center')
#     else:
#         plt.legend(loc='upper left')
#
#     plt.tight_layout()
#     plt.savefig('{0}/overall-{1}.pdf'.format(plot_save_path, in_out_label[i_degree]), format='pdf')
#     plt.clf()

# Results pickle files are in the following order
#   local-formed-in-degree, global-formed-in-degree, local-formed-out-degree, global-formed-out-degree
#   local-not-formed-in-degree, global-not-formed-in-degree, local-not-formed-out-degree, global-not-formed-out-degree

triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/cdf-plots'

gl_labels = ['Local', 'Global']
z = 1.96


pos = list(range(len(triangle_types)))
bar_width = 0.20
opacity = 0.6
error_config = {'ecolor': '0.3'}
bar_legends = ['In-degree Formed', 'In-degree Not Formed', 'Out-degree Formed', 'Out-degree Not Formed']
dif_results = ['id-formed', 'id-not-formed', 'od-formed', 'od-not-formed']
bar_color = ['r', 'b', 'g', 'y']


all_results = {
    'Global': {
        'id-formed': [],
        'id-not-formed': [],
        'od-formed': [],
        'od-not-formed': [],
        'id-formed-err': [],
        'id-not-formed-err': [],
        'od-formed-err': [],
        'od-not-formed-err': []
    },
    'Local': {
        'id-formed': [],
        'id-not-formed': [],
        'od-formed': [],
        'od-not-formed': [],
        'id-formed-err': [],
        'id-not-formed-err': [],
        'od-formed-err': [],
        'od-not-formed-err': []
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

    for i in range(2):
        # computing mean
        all_results[gl_labels[i]]['id-formed'].append(np.mean(results[:, i]))
        all_results[gl_labels[i]]['id-not-formed'].append(np.mean(results[:, i + 4]))
        all_results[gl_labels[i]]['od-formed'].append(np.mean(results[:, i + 2]))
        all_results[gl_labels[i]]['od-not-formed'].append(np.mean(results[:, i + 6]))

        # computing 95% confidence interval
        all_results[gl_labels[i]]['id-formed-err'].append(get_mean_ci(results[:, i], z))
        all_results[gl_labels[i]]['id-not-formed-err'].append(get_mean_ci(results[:, i + 4], z))
        all_results[gl_labels[i]]['od-formed-err'].append(get_mean_ci(results[:, i + 2], z))
        all_results[gl_labels[i]]['od-not-formed-err'].append(get_mean_ci(results[:, i + 6], z))

    print(triangle_types[t] + ": Done")

# plotting
for i_degree in range(2):
    fig, ax = plt.subplots()

    for i_bar in range(len(dif_results)):
        plt.bar(np.arange(len(triangle_types)) + bar_width * i_bar,
                all_results[gl_labels[i_degree]][dif_results[i_bar]],
                bar_width,
                alpha=opacity,
                color=bar_color[i_bar],
                yerr=all_results[gl_labels[i_degree]][dif_results[i_bar] + '-err'],
                error_kw=error_config,
                label=bar_legends[i_bar])

    plt.xlabel('Triangle Types')
    plt.ylabel('Mean Normalized {0} Degree of Common Neighbors'.format(gl_labels[i_degree]))
    plt.xticks(np.arange(len(triangle_types)) + bar_width * 1.5, triangle_types)

    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('{0}/overall-{1}.pdf'.format(plot_save_path, gl_labels[i_degree]), format='pdf')
    plt.clf()