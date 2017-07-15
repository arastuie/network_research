import pickle
import numpy as np
import helpers as h
import networkx as nx
from joblib import Parallel, delayed
import os
import degree_based_analysis as a1
import num_cluster_per_cn_analysis as a2
import tot_num_cluster_of_all_cn_analysis as a3
import tot_num_cluster_based_num_cn_analysis as a4
import percent_of_cn_belonging_to_same_cluster as a5
import hop_degree_analysis as a6
#import gplus_hop_degree_directed_analysis_t06 as a7
import gplus_hop_degree_directed_analysis as a7


# # reading FB graph
# original_graph = h.read_facebook_graph()
#
# # extracting ego centric networks
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_fb(original_graph, 50, "random_50_ego_nets.pckl",
#                                                                    search_type='random', hop=2, center=True)
#
# print("Loading Facebook random 200 ego centric networks...")
#
# with open('../Data/random_200_ego_nets.pckl', 'rb') as f:
#     ego_centric_networks, ego_nodes = pickle.load(f)

print("Analysing ego centric networks...")

# num_cluster = 8

# for o in range(len(ego_centric_networks)):
#     print("EgoNet #%d" % o)
#
#     a1.run_degree_based_analysis(ego_centric_networks[o], ego_nodes[o], o, False, '../Plots/degree_based')
#
#     a2.run_num_cluster_per_cn_analysis(ego_centric_networks[o], ego_nodes[o], num_cluster, o, True,
#                                        '../Plots/cluster_per_node')
#
#     a3.run_tot_num_cluster_of_all_cn_analysis(ego_centric_networks[o], ego_nodes[o], num_cluster, o, True,
#                                               '../Plots/total_cluster/6-2-17-13-30')
#
#     a4.run_tot_num_cluster_based_num_cn_analysis(ego_centric_networks[o], ego_nodes[o], num_cluster, o, True,
#                                                  '../Plots/total_cluster_overall')
#
#     a5.run_percent_of_cn_belonging_to_same_cluster(ego_centric_networks[o], ego_nodes[o], num_cluster, o, True,
#                                                    '../Plots/percent_of_cn_belonging_to_same_cluster')
#
#     a6.run_hop_degree_analysis(ego_centric_networks[o], ego_nodes[o], o, True, '../Plots/gplus_hop_degree_based')
#
# Parallel(n_jobs=20)(delayed(a6.gplus_run_hop_degree_analysis)(ego_node_file, True, '../Plots/gplus_hop_degree_based')
#                     for ego_node_file in os.listdir('../Data/gplus-ego/first-hop-nodes'))

# Parallel(n_jobs=20)(delayed(a6.run_hop_degree_analysis)
#                     (ego_centric_networks[o], ego_nodes[o], o, True, '../Plots/hop_degree_based')
#                     for o in range(len(ego_centric_networks)))

# Parallel(n_jobs=20)(delayed(a7.gplus_run_hop_degree_directed_analysis)
#                     (ego_node_file, True, '../Plots/gplus_hop_degree_based')
#                     for ego_node_file in os.listdir('../Data/gplus-ego/first-hop-nodes'))

triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']


def test_directed_triangle(triangle_type):
    overall_means = {
        'formed_in_degree_first_hop': [],
        'not_formed_in_degree_first_hop': [],
        'formed_in_degree_second_hop': [],
        'not_formed_in_degree_second_hop': [],
        'formed_out_degree_first_hop': [],
        'not_formed_out_degree_first_hop': [],
        'formed_out_degree_second_hop': [],
        'not_formed_out_degree_second_hop': [],
    }

    for ego_net_file in os.listdir('../Data/gplus-ego/first-hop-nodes'):
        a7.gplus_run_hop_degree_directed_analysis('../Data/gplus-ego/first-hop-nodes/%s' % ego_net_file, triangle_type,
                                                  overall_means, True, '../Plots/hop_degree_based')

    with open('../Plots/hop_degree_based/{0}/overall.txt'.format(triangle_type), 'w') as info_file:
        info_file.write("OVERALL SCORES:\n")
        info_file.write("In-degree First Hop:\n\tFEM:{0:.3f} \t NFEM:{1:.3f}\n\n"
                        .format(np.mean(overall_means['formed_in_degree_first_hop']),
                                np.mean(overall_means['not_formed_in_degree_first_hop'])))

        info_file.write("In-degree Second Hop:\n\tFEM:{0:.3f} \t NFEM:{1:.3f}\n\n"
                        .format(np.mean(overall_means['formed_in_degree_second_hop']),
                                np.mean(overall_means['not_formed_in_degree_second_hop'])))

        info_file.write("Out-degree First Hop:\n\tFEM:{0:.3f} \t NFEM:{1:.3f}\n\n"
                        .format(np.mean(overall_means['formed_out_degree_first_hop']),
                                np.mean(overall_means['not_formed_out_degree_first_hop'])))

        info_file.write("Out-degree Second Hop:\n\tFEM:{0:.3f} \t NFEM:{1:.3f}\n\n"
                        .format(np.mean(overall_means['formed_out_degree_second_hop']),
                                np.mean(overall_means['not_formed_out_degree_second_hop'])))

    print("Done with {0} triangle type".format(triangle_type))

Parallel(n_jobs=9)(delayed(test_directed_triangle)(tri_type) for tri_type in triangle_types)
