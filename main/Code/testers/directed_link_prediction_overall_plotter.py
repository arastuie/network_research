import pickle
import numpy as np
import matplotlib.pyplot as plt

# LP results pickle files are in the following order
#   cn, dccn_i, dccn_o, aa_i, aa_o, dcaa_i, dcaa_o


triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
plotting_k_vals = [3, 10, 20, 30]
plotting_k_vals_index = [1, 3, 5, 7]

pos = list(range(len(triangle_types)))
bar_width = 0.10
opacity = 0.6
error_config = {'ecolor': '0.3', 'elinewidth': '0.1'}
bar_color = ['r', 'b', 'g', '0.5', 'm', 'c', 'black', 'y']

plot_labels = ['DCCN VS CN', 'DCAA VS AA']
plot_labels_names = ['gplus-lp-cn-overall', 'gplus-lp-aa-overall']

plot_order = [['imp_cn_i', 'imp_cn_o'], ['imp_aa_i', 'imp_aa_o']]
plot_legends = ['In-degree K = {0}', 'Out-degree K = {0}']
all_imp = {
    'imp_cn_i': {},
    'imp_cn_o': {},
    'imp_aa_i': {},
    'imp_aa_o': {},
}

all_imp_err = {
    'imp_cn_i_err': {},
    'imp_cn_o_err': {},
    'imp_aa_i_err': {},
    'imp_aa_o_err': {}
}

for key in all_imp.keys():
    for kv in plotting_k_vals_index:
        all_imp[key][top_k_values[kv]] = []
        all_imp_err[key + '_err'][top_k_values[kv]] = []

for triangle_type in triangle_types:
    with open(result_file_base_path + 'plot_ready_data/' + triangle_type + '.pckle', 'rb') as f:
        imp_mse = pickle.load(f)

    for key in all_imp.keys():
        for kv in plotting_k_vals_index:
            all_imp[key][top_k_values[kv]].append(imp_mse[key][kv])
            all_imp_err[key + '_err'][top_k_values[kv]].append(imp_mse[key + '_err'][kv])

print("Data Loaded!")

# plotting

for order_i in range(len(plot_order)):
    fig, ax = plt.subplots()

    for i in range(len(plot_order[order_i])):
        for ki in range(len(plotting_k_vals)):
            index = ki + (i * 4)
            plt.bar(np.arange(len(triangle_types)) + bar_width * index,
                    all_imp[plot_order[order_i][i]][plotting_k_vals[ki]],
                    bar_width,
                    alpha=opacity,
                    color=bar_color[index],
                    yerr=all_imp_err[plot_order[order_i][i] + '_err'][plotting_k_vals[ki]],
                    error_kw=error_config,
                    edgecolor='black',
                    linewidth='0',
                    label=plot_legends[i].format(plotting_k_vals[ki]))

    plt.xlabel('Triangle Types')
    plt.ylabel('{0} Percent Improvement'.format(plot_labels[order_i]))
    plt.xticks(np.arange(len(triangle_types)) + bar_width * 4, triangle_types)

    plt.legend(loc='upper right',
               ncol=2, mode="expand", borderaxespad=0.)

    if order_i == 0:
        plt.ylim(ymax=6)
    else:
        plt.ylim(ymax=4)

    plt.tight_layout()
    plt.savefig('{0}/{1}.pdf'.format(plot_save_path, plot_labels_names[order_i]), format='pdf')
    plt.clf()

