import pickle
import numpy as np
import matplotlib.pyplot as plt
import digg_net_helpers as dh

plot_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/digg/plots'
path = '/shared/Results/EgocentricLinkPrediction/main/empirical/digg/pickle-files'

names = ['Global Degree', 'Local Degree']
bar_width = 0.40
opacity = 0.6
z = 1.96
error_config = {'ecolor': '0.3', 'capsize': 4, 'lw': 3, 'capthick': 2}
bar_legends = ['Formed', 'Not Formed']
dif_results_for_plotting = ['formed', 'not-formed']
bar_color = ['r', 'b']

print("Reading Result file")

with open('{0}/{1}'.format(path, '90-days-duration-results.pckle'), 'rb') as f:
    all_results = pickle.load(f)

print("Results in...")

plot_results = dh.extract_empirical_overall_plotter_data(all_results, z)

print("plot results extracted...")

plt.rcParams["figure.figsize"] = (8, 10)

# plotting
plt.rc('legend', fontsize=25)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=14)

fig, ax = plt.subplots()
degrees = ['global-formed', 'global-not-formed', 'local-formed', 'local-not-formed']

plt.bar(np.arange(4), [plot_results[x] for x in degrees],
            bar_width,
            alpha=opacity,
            yerr=[plot_results['error'][x] for x in degrees],
            error_kw=error_config,
            label=degrees)

# print(plot_results)
# for i_bar in range(len(degrees)):
#     plt.bar(np.arange(len(names)) + bar_width * i_bar,
#             [plot_results[x] for x in degrees[i_bar]],
#             bar_width,
#             alpha=opacity,
#             color=bar_color[i_bar],
#             yerr=[plot_results['error'][x] for x in degrees[i_bar]],
#             error_kw=error_config,
#             label=bar_legends)

plt.ylabel('Mean Normalized Degree', fontsize=25)
# plt.xticks(np.arange(len(names)) + bar_width / 2, names)
plt.legend(loc='upper left')

plt.savefig('{0}/overall.pdf'.format(plot_path), format='pdf')
plt.clf()
