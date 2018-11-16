import os
import pickle
import numpy as np
import helpers as h
import networkx as nx
import digg_helpers as digg
import gplus_helpers as gplus
import flickr_helpers as flickr
import matplotlib.pyplot as plt
import directed_graphs_helpers as dgh


def plot_all_degree_distributions(data, plot_save_path):
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    local_in, local_out, global_in, global_out = data

    # in degree
    global_in = np.array(global_in) + 1
    global_in_count = np.unique(global_in, return_counts=True)
    global_in_count = h.log_binning(global_in_count, 75)

    local_in = np.array(local_in) + 1
    local_in_count = np.unique(local_in, return_counts=True)
    local_in_count = h.log_binning(local_in_count, 75)

    plt.xscale('log')
    plt.yscale('log')

    plt.scatter(local_in_count[0], local_in_count[1], c='b', marker='x', alpha=0.9, label="Personalized")
    plt.scatter(global_in_count[0], global_in_count[1], c='r', marker='*', alpha=0.9, label="Global")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="best", fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xlabel('In-degree', fontsize=20)
    plt.tight_layout()
    plt.savefig('{}in-degree-dist.pdf'.format(plot_save_path), format='pdf')
    # plt.show()
    plt.clf()

    # out degree
    global_out = np.array(global_out) + 1
    global_out_count = np.unique(global_out, return_counts=True)
    global_out_count = h.log_binning(global_out_count, 100)

    local_out = np.array(local_out) + 1
    local_out_count = np.unique(local_out, return_counts=True)
    local_out_count = h.log_binning(local_out_count, 100)

    plt.xscale('log')
    plt.yscale('log')

    plt.scatter(local_out_count[0], local_out_count[1], c='b', marker='x', alpha=0.9, label="Personalized")
    plt.scatter(global_out_count[0], global_out_count[1], c='r', marker='*', alpha=0.9, label="Global")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="best", fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xlabel('Out-degree', fontsize=20)
    plt.tight_layout()
    plt.savefig('{}out-degree-dist.pdf'.format(plot_save_path), format='pdf')
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    ## Local Degree Distribution Plots
    # G+
    # gplus_data = dgh.get_degree_dist_data(gplus.local_degree_empirical_results_base_path +
    #                                       'local-degree-dist/pickle-files-2/', gather_individual_results=False,
    #                                       get_z_id_global_degree=True)
    # plot_all_degree_distributions(gplus_data, gplus.local_degree_empirical_results_base_path +
    #                               'local-degree-dist/plots-2/')
    #
    # # Flickr
    # flickr_data = dgh.get_degree_dist_data(flickr.local_degree_dist_results_base_path + 'pickle-files-2/',
    #                                        gather_individual_results=False, get_z_id_global_degree=True)
    # plot_all_degree_distributions(flickr_data, flickr.local_degree_dist_results_base_path + 'plots-2/')

    # Digg
    digg_data = dgh.get_degree_dist_data(digg.directed_local_degree_empirical_base_results_path +
                                         'local-degree-dist/pickle-files-1/', gather_individual_results=False,
                                         get_z_id_global_degree=False)
    digg_graph = digg.read_graph_as_directed()
    in_degree = []
    out_degree = []
    for n in digg_graph.nodes:
        in_degree.append(digg_graph.in_degree(n))
        out_degree.append(digg_graph.out_degree(n))
    digg_data = [digg_data[0], digg_data[1], in_degree, out_degree]
    plot_all_degree_distributions(digg_data, digg.directed_local_degree_empirical_base_results_path +
                                  'local-degree-dist/plots-1/')