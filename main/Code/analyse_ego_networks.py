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
import gplus_hop_degree_analysis_2 as a6


# # reading FB graph
# original_graph = h.read_facebook_graph()
#
# # extracting ego centric networks
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_fb(original_graph, 50, "random_50_ego_nets.pckl",
#                                                                    search_type='random', hop=2, center=True)

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

Parallel(n_jobs=15)(delayed(a6.gplus_run_hop_degree_analysis)(ego_node_file, True, '../Plots/gplus_hop_degree_based_2') for ego_node_file in os.listdir('../Data/gplus-ego'))
