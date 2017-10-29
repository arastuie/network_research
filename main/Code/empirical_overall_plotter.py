import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_mean_ci(res, z_value):
    return z_value * np.std(res) / np.sqrt(len(res))


# plot_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/cdf'
# path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb'
#
# paths = {
#     'Global': ['global/lower-6', 'global/after-6'],
#     'Local': ['local/lower-6', 'local/after-6']
# }
#
# dif_res = {
#     'Global': ['global-formed', 'global-not-formed'],
#     'Local': ['local-formed', 'local-not-formed']
# }
# names = ['Before PYMK', 'After PYMK']
#
# gl_labels = ['Local', 'Global']
# z = 1.96
#
#
# pos = list(range(len(names)))
# bar_width = 0.20
# opacity = 0.6
# error_config = {'ecolor': '0.3'}
# bar_legends = ['Global Formed', 'Global Not Formed', 'Local Formed', 'Local Not Formed']
# dif_results_for_plotting = ['global-formed', 'global-not-formed', 'local-formed', 'local-not-formed']
# bar_color = ['r', 'b', 'g', 'y']
#
# all_results = {
#     'global-formed': [],
#     'global-not-formed': [],
#     'local-formed': [],
#     'local-not-formed': [],
#     'global-formed-err': [],
#     'global-not-formed-err': [],
#     'local-formed-err': [],
#     'local-not-formed-err': []
# }
#
# for ps in paths:
#     for i in range(len(paths[ps])):
#         with open('{0}/pickle-files/{1}/temp/total-result.pckl'.format(path, paths[ps][i]), 'rb') as f:
#             res = pickle.load(f)
#
#         for ii in range(2):
#             all_results[dif_res[ps][ii]].append(np.mean(res[ii]))
#             all_results[dif_res[ps][ii] + '-err'].append(get_mean_ci(res[ii], z))
#
# # plotting
#
# fig, ax = plt.subplots()
#
# for i_bar in range(len(dif_results_for_plotting)):
#     plt.bar(np.arange(len(names)) + bar_width * i_bar,
#             all_results[dif_results_for_plotting[i_bar]],
#             bar_width,
#             alpha=opacity,
#             color=bar_color[i_bar],
#             yerr=all_results[dif_results_for_plotting[i_bar] + '-err'],
#             error_kw=error_config,
#             label=bar_legends[i_bar])
#
# plt.ylabel('Mean Normalized Degree of Common Neighbors')
# plt.xticks(np.arange(len(names)) + bar_width * 1.5, names)
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('{0}/overall.pdf'.format(plot_path), format='pdf')
# plt.clf()


plot_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/cdf'
path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb'

paths = {
    'Global': ['global/lower-6', 'global/after-6'],
    'Local': ['local/lower-6', 'local/after-6']
}

dif_res = {
    'Global': ['global-formed', 'global-not-formed'],
    'Local': ['local-formed', 'local-not-formed']
}
names = ['Before PYMK', 'After PYMK']

gl_labels = ['local', 'global']
gl_labels_plotting = ['Local', 'Global']
z = 1.96


pos = list(range(len(names)))
bar_width = 0.40
opacity = 0.6
error_config = {'ecolor': '0.3'}
bar_legends = ['Formed', 'Not Formed']
dif_results_for_plotting = ['formed', 'not-formed']
bar_color = ['r', 'b', 'g', 'y']

all_results = {
    'global-formed': [],
    'global-not-formed': [],
    'local-formed': [],
    'local-not-formed': [],
    'global-formed-err': [],
    'global-not-formed-err': [],
    'local-formed-err': [],
    'local-not-formed-err': []
}

for ps in paths:
    for i in range(len(paths[ps])):
        with open('{0}/pickle-files/{1}/temp/total-result.pckl'.format(path, paths[ps][i]), 'rb') as f:
            res = pickle.load(f)

        for ii in range(2):
            all_results[dif_res[ps][ii]].append(np.mean(res[ii]))
            all_results[dif_res[ps][ii] + '-err'].append(get_mean_ci(res[ii], z))


plt.rcParams["figure.figsize"] = (8, 10)

# plotting
for i_degree in range(2):
    plt.rc('legend', fontsize=25)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=14)

    fig, ax = plt.subplots()

    for i_bar in range(len(dif_results_for_plotting)):
        plt.bar(np.arange(len(names)) + bar_width * i_bar,
                all_results[gl_labels[i_degree] + '-' + dif_results_for_plotting[i_bar]],
                bar_width,
                alpha=opacity,
                color=bar_color[i_bar],
                yerr=all_results[gl_labels[i_degree] + '-' + dif_results_for_plotting[i_bar] + '-err'],
                error_kw=error_config,
                label=bar_legends[i_bar])

    plt.ylabel('Mean Normalized {0} Degree of Common Neighbors'.format(gl_labels_plotting[i_degree]), fontsize=25)
    plt.xticks(np.arange(len(names)) + bar_width / 2, names)
    plt.legend(loc='upper left')
    # plt.tight_layout()
    if i_degree == 1:
        plt.ylim(ymax=0.1)

    plt.savefig('{0}/overall-{1}.pdf'.format(plot_path, gl_labels[i_degree]), format='pdf')
    plt.clf()
