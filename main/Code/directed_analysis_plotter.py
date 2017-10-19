import os
import math
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

gl_labels = ['Local', 'Global']

for triangle_type in triangle_types:
    results = []

    # loading result data
    for result_file in os.listdir(result_file_base_path + triangle_type):
        with open(result_file_base_path + triangle_type + '/' + result_file, 'rb') as f:
            egonet_result = pickle.load(f)

        results.append(egonet_result)

    results = np.array(results)

    # plotting
    for i in range(0, 2):
        plt.rc('legend', fontsize=12)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)

        h.add_ecdf_with_bond_plot(results[:, i], 'Indegree Formed Edges', 'r')
        h.add_ecdf_with_bond_plot(results[:, i + 4], 'Indegree Not Formed Edges', 'b')

        h.add_ecdf_with_bond_plot(results[:, i + 2], 'Outdegree Formed Edges', 'y')
        h.add_ecdf_with_bond_plot(results[:, i + 6], 'Outdegree Not Formed Edges', 'g')

        plt.ylabel('Empirical CDF', fontsize=14)
        plt.xlabel('Mean Normalized {0} Degree'.format(gl_labels[i]), fontsize=14)
        plt.legend(loc='lower right')
        plt.suptitle('Number of egonets analyzed: {0}'.format(len(results)))
        current_fig = plt.gcf()
        # current_fig.savefig('{0}/{1}-{2}-cdf.pdf'.format(plot_save_path, triangle_type, gl_labels[i]), format='pdf')
        current_fig.savefig('{0}/{1}-{2}-cdf.png'.format(plot_save_path, triangle_type, gl_labels[i]))
        plt.clf()

    print(triangle_type + ": Done")

print("Done!")


