import networkx as nx
import helpers as h
import GSC.generalized_spectral_clustering as gsc
import numpy as np
import pickle

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
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_gplus(graph, 200, "random_200_gplus_ego_nets.pckl",
#                                                                       search_type='random', hop=2, center=True)

ego_centric_networks = h.read_gplus_ego_graph(200)

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

# h.gplus_get_all_nodes_first_appeared(0)
