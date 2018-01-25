import networkx as nx
import helpers as h
import GSC.generalized_spectral_clustering as gsc
import numpy as np
import pickle
import gplus_hop_degree_directed_analysis as directed_analysis
from joblib import Parallel, delayed
import os
import sys
import directed_graphs_helpers as dh

# graph = nx.read_gml("../Data/karate.gml")
#
# gsc_model = gsc.gsc(graph, 3)
# node_list, clusters = gsc_model.get_clusters(kmedian_max_iter=100, num_cluster_per_node=False)
#
# print(h.get_cluster_overlap(clusters))
#
# print(h.get_cluster_coefficient(clusters))
#print(node_list)
#print(clusters)

# hashtable = {2: [6, 4], 5: [9, 14], 1: [2, 9]}
# hashtable2 = {2: [6, 4], 5: [9, 14], 0: [4, 0]}
#
# a, b = h.keep_common_keys(hashtable, hashtable2)
# print(len(a))
# print(b)
#
# for key in hashtable:
#     print(type(key))

# graph = h.read_gplus_graph()
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_gplus(graph, 400, "random_400_gplus_ego_nets.pckl",
#                                                                       search_type='random', hop=2, center=True)


# graph = h.read_facebook_graph()
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_fb(graph, 500, 'random_500_ego_nets.pckl',
#                                                                    search_type='random', hop=2, center=True)


# ego_centric_networks = h.read_gplus_ego_graph(200)

# with open('../Data/gplus-nodes-list.pckl', 'rb') as f:
#     nodes = pickle.load(f)
#
# print(len(nodes))
# print(0)
# with open("/shared/DataSets/GooglePlus_Gong2012/gplus/imc12/direct_social_structure.txt") as infile:
#     all_edges = []
#     cnt = 0
#     print(cnt, end='\r')
#     for l in infile:
#         if cnt % 100000 == 0:
#             print(cnt, end='\r')
#
#         nums = l.split(" ")
#         if nums[2][0] == '0':
#             all_edges.append((int(nums[0]), int(nums[1]), {'snapshot': 0}))
#
#         cnt += 1
#
#     with open('../Data/gplus-edges-snap-0-list.pckl', 'wb') as f:
#         pickle.dump(all_edges, f, protocol=-1)
#
# print(1)
# with open("/shared/DataSets/GooglePlus_Gong2012/gplus/imc12/direct_social_structure.txt") as infile:
#     all_edges = []
#     cnt = 0
#     print(cnt, end='\r')
#     for l in infile:
#         if cnt % 100000 == 0:
#             print(cnt, end='\r')
#
#         nums = l.split(" ")
#         if nums[2][0] == '1':
#             all_edges.append((int(nums[0]), int(nums[1]), {'snapshot': 1}))
#
#         cnt += 1
#
#     with open('../Data/gplus-edges-snap-1-list.pckl', 'wb') as f:
#         pickle.dump(all_edges, f, protocol=-1)
#
# print(2)
# with open("/shared/DataSets/GooglePlus_Gong2012/gplus/imc12/direct_social_structure.txt") as infile:
#     all_edges = []
#     cnt = 0
#     print(cnt, end='\r')
#     for l in infile:
#         if cnt % 100000 == 0:
#             print(cnt, end='\r')
#
#         nums = l.split(" ")
#         if nums[2][0] == '2':
#             all_edges.append((int(nums[0]), int(nums[1]), {'snapshot': 2}))
#
#         cnt += 1
#
#     with open('../Data/gplus-edges-snap-2-list.pckl', 'wb') as f:
#         pickle.dump(all_edges, f, protocol=-1)
#
# print(3)
# with open("/shared/DataSets/GooglePlus_Gong2012/gplus/imc12/direct_social_structure.txt") as infile:
#     all_edges = []
#     cnt = 0
#     print(cnt, end='\r')
#     for l in infile:
#         if cnt % 100000 == 0:
#             print(cnt, end='\r')
#
#         nums = l.split(" ")
#         if nums[2][0] == '3':
#             all_edges.append((int(nums[0]), int(nums[1]), {'snapshot': 3}))
#
#         cnt += 1
#
#     with open('../Data/gplus-edges-snap-3-list.pckl', 'wb') as f:
#         pickle.dump(all_edges, f, protocol=-1)

# h.gplus_get_all_nodes_appeared_in_snapshot(0)

# overall_means = {
#     'formed_in_degree_first_hop': [],
#     'not_formed_in_degree_first_hop': [],
#     'formed_in_degree_second_hop': [],
#     'not_formed_in_degree_second_hop': [],
#     'formed_out_degree_first_hop': [],
#     'not_formed_out_degree_first_hop': [],
#     'formed_out_degree_second_hop': [],
#     'not_formed_out_degree_second_hop': [],
# }
#
# # directed_analysis.gplus_run_hop_degree_directed_analysis('862541.pckle', 'T01', overall_means, False, '../Plots/hop_degree_based')
# Parallel(n_jobs=2)(delayed(directed_analysis.gplus_run_hop_degree_directed_analysis)
#                     ('../Data/gplus-ego/%s' % ego_net_file, 'T01', overall_means, True,
#                      '../Plots/hop_degree_based') for ego_net_file in os.listdir('../Data/gplus-ego'))
#
# print(len(overall_means['formed_in_degree_first_hop']))

# with open('../Data/gplus/gplus-nodes-list.pckl', 'rb') as f:
#     all_nodes = pickle.load(f)
#     print(len(all_nodes))
#
#
# fb_graph = h.read_facebook_graph()
# new = 0
# old = 1000000000000
# for u, v, t in fb_graph.edges_iter(data='timestamp', default=-1):
#     if new < t:
#         new = t
#
#     if old > t and t != -1:
#         old = t
#
# print(new)
# print(old)
#
# def get_ego_centric_networks_in_fb(original_graph):
#     orig_snapshots = []
#
#     oldest_timestamp = 1157454929
#     seconds_in_90_days = 7776000
#     for i in range(10):
#         orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in original_graph.edges(data=True)
#                                         if d['timestamp'] < (oldest_timestamp + seconds_in_90_days * i)]))
#
#     orig_snapshots.append(original_graph)
#     orig_nodes = nx.nodes(orig_snapshots[0])
#
#     ego_nodes_split = np.array_split(np.array(orig_nodes), 28)
#     Parallel(n_jobs=28)(delayed(save_fb_egonet)(orig_snapshots, ego_nodes_split[i], i) for i in range(0, 28))
#
#     return
#
#
# def save_fb_egonet(orig_snapshots, ego_nodes, index):
#     for node in ego_nodes:
#         ego_centric_network_snapshots = []
#         for i in range(len(orig_snapshots)):
#             ego_centric_network_snapshots.append(nx.ego_graph(orig_snapshots[i], node, radius=2, center=True))
#
#         with open('/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}/{1}.pckl'.format(index, node), 'wb') as f:
#             pickle.dump([ego_centric_network_snapshots, node], f, protocol=-1)
#
#
# graph = h.read_facebook_graph()
# get_ego_centric_networks_in_fb(graph)

# dh.read_gplus_ego_graph(100000)
# dh.create_gplus_multiple_egonets(10000, 55)

# result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/'
# empirical_analyzed_egonets_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files/'
# for tt in ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']:
#     p = len(os.listdir(result_file_base_path + tt)) / len(os.listdir(empirical_analyzed_egonets_path + tt))
#     print("{0} is {1:.1f}% complete!".format(tt, p * 100))


# dh.read_entire_gplus_network()
# dh.read_gplus_ego_graph(2)

dh.read_ego_gplus_graph_by_batch_parallelizer(50)

# data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/'
#
# over_50 = []
# over_100 = []
# over_150 = []
# over_200 = []
# over_300 = []
#
# n = 0
# print("")
#
# for ego_net_file in os.listdir(data_file_base_path):
#     with open(data_file_base_path + ego_net_file, 'rb') as f:
#         ego_node, ego_net = pickle.load(f)
#
#     cnt = nx.number_of_nodes(ego_net)
#     # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
#     if cnt > 300000:
#         over_300.append(ego_node)
#     elif cnt > 200000:
#         over_200.append(ego_node)
#     elif cnt > 150000:
#         over_150.append(ego_node)
#     elif cnt > 100000:
#         over_100.append(ego_node)
#     elif cnt > 50000:
#         over_50.append(ego_node)
#
#     n += 1
#     if n % 100 == 0:
#         print("Num egonets checked: {0}".format(n), end='\r')
#
# over_50 = over_50 + over_100 + over_150 + over_200 + over_300
# over_100 = over_100 + over_150 + over_200 + over_300
# over_150 = over_150 + over_200 + over_300
# over_200 = over_200 + over_300
#
# print("Number of egonets with over {0}K nodes: {1}".format(50, len(over_50)))
# print("Number of egonets with over {0}K nodes: {1}".format(100, len(over_100)))
# print("Number of egonets with over {0}K nodes: {1}".format(150, len(over_150)))
# print("Number of egonets with over {0}K nodes: {1}".format(200, len(over_200)))
# print("Number of egonets with over {0}K nodes: {1}".format(300, len(over_300)))
#
# with open(data_file_base_path + 'egonets-info/' + 'set_of_egonets_with_over_50K_nodes.pckle', 'wb') as f:
#     pickle.dump(set(over_50), f, protocol=-1)
#
# with open(data_file_base_path + 'egonets-info/' + 'set_of_egonets_with_over_100K_nodes.pckle', 'wb') as f:
#     pickle.dump(set(over_100), f, protocol=-1)
#
# with open(data_file_base_path + 'egonets-info/' + 'set_of_egonets_with_over_150K_nodes.pckle', 'wb') as f:
#     pickle.dump(set(over_150), f, protocol=-1)
#
# with open(data_file_base_path + 'egonets-info/' + 'set_of_egonets_with_over_200K_nodes.pckle', 'wb') as f:
#     pickle.dump(set(over_200), f, protocol=-1)
#
# with open(data_file_base_path + 'egonets-info/' + 'set_of_egonets_with_over_300K_nodes.pckle', 'wb') as f:
#     pickle.dump(set(over_300), f, protocol=-1)