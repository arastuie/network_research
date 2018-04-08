import pickle
import numpy as np
from joblib import Parallel, delayed
import os
import Code.degree_based_analysis as a1
import Code.num_cluster_per_cn_analysis as a2
import Code.tot_num_cluster_of_all_cn_analysis as a3
import Code.tot_num_cluster_based_num_cn_analysis as a4
import Code.percent_of_cn_belonging_to_same_cluster as a5
import Code.hop_degree_analysis as a6
#import Code.gplus_hop_degree_directed_analysis_t06 as a7
import Code.gplus_hop_degree_directed_analysis as a7
import Code.adamic_adar_analysis as a8
import matplotlib.pyplot as plt

print("Analysing ego centric networks...")
path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/pickle-files/local/lower-6/temp'


def run_analysis(index):
    mfems = []
    mnfems = []

    cnt = 0

    for ego_net_file in os.listdir('/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}'.format(index)):
        with open('/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}/{1}'.format(index, ego_net_file), 'rb') as f:
            egonet_snapshots, ego_node = pickle.load(f)

        mfem, mnfem = a6.run_hop_local_degree_analysis(egonet_snapshots, ego_node, range(0, 5))
        # mfem, mnfem = a8.run_hop_global_degree_analysis(egonet_snapshots, ego_node, 0, False, '../Plots/global_degree')

        if mfem != -1:
            mfems.append(mfem)
            mnfems.append(mnfem)

        cnt = cnt + 1

        if cnt % 500 == 0:
            print("Index {0} -> {1}".format(index, cnt))

    if len(mfems) > 0:
        print("mfem -> {0}".format(np.mean(mfems)))
        print("mnfem -> {0}".format(np.mean(mnfems)))

        with open('{0}/{1}-egonodes-result.pckl'.format(path, index), 'wb') as f:
            pickle.dump([mfems, mnfems], f, protocol=-1)
    else:
        print("No analysis in index {0}".format(index))


Parallel(n_jobs=28)(delayed(run_analysis)(i) for i in range(0, 28))

print("Merging all files...")

mfems = []
mnfems = []
for result_file in os.listdir(path):
    with open('{0}/{1}'.format(path, result_file), 'rb') as f:
        mf, mn = pickle.load(f)

    mfems = mfems + mf
    mnfems = mnfems + mn

with open('{0}/total-result.pckl'.format(path), 'wb') as f:
    pickle.dump([mfems, mnfems], f, protocol=-1)

### Reading analyzed files
# with open('{0}/total-result.pckl'.format(path), 'rb') as f:
#     mfems, mnfems = pickle.load(f)

plot_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/pickle-files/local/lower-6/'

plt.hist(mfems, color='r', alpha=0.8, weights=np.zeros_like(mfems) + 1. / len(mfems),
         label="MFEM: {0:.2f}".format(np.mean(mfems)))

plt.hist(mnfems, color='b', alpha=0.5, weights=np.zeros_like(mnfems) + 1. / len(mnfems),
         label="MNFEM: {0:.2f}".format(np.mean(mnfems)))

plt.legend(loc='upper right')
plt.ylabel('Relative Frequency')
plt.xlabel('Mean Normalized Local Degree of Common Neighbors')

current_fig = plt.gcf()
current_fig.savefig('{0}/overall-mean-normal.eps'.format(plot_path), format='eps', dpi=1000)
current_fig.savefig('{0}/overall-mean-normal-png.png'.format(plot_path))

print("Number of egonets analyzed: {0}".format(len(mfems)))
print("Done")



######### old code for clustering ###########
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
#
# triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
#
#
# def test_directed_triangle(triangle_type):
#     overall_means = {
#         'formed_in_degree_first_hop': [],
#         'not_formed_in_degree_first_hop': [],
#         'formed_in_degree_second_hop': [],
#         'not_formed_in_degree_second_hop': [],
#         'formed_out_degree_first_hop': [],
#         'not_formed_out_degree_first_hop': [],
#         'formed_out_degree_second_hop': [],
#         'not_formed_out_degree_second_hop': [],
#     }
#
#     for ego_net_file in os.listdir('../Data/gplus-ego/first-hop-nodes'):
#         a7.gplus_run_hop_degree_directed_analysis('../Data/gplus-ego/first-hop-nodes/%s' % ego_net_file, triangle_type,
#                                                   overall_means, True, '../Plots/hop_degree_based')
#
#     with open('../Plots/hop_degree_based/{0}/overall.txt'.format(triangle_type), 'w') as info_file:
#         info_file.write("OVERALL SCORES:\n")
#         info_file.write("In-degree First Hop:\n\tFEM:{0:.3f} \t NFEM:{1:.3f}\n\n"
#                         .format(np.mean(overall_means['formed_in_degree_first_hop']),
#                                 np.mean(overall_means['not_formed_in_degree_first_hop'])))
#
#         info_file.write("In-degree Second Hop:\n\tFEM:{0:.3f} \t NFEM:{1:.3f}\n\n"
#                         .format(np.mean(overall_means['formed_in_degree_second_hop']),
#                                 np.mean(overall_means['not_formed_in_degree_second_hop'])))
#
#         info_file.write("Out-degree First Hop:\n\tFEM:{0:.3f} \t NFEM:{1:.3f}\n\n"
#                         .format(np.mean(overall_means['formed_out_degree_first_hop']),
#                                 np.mean(overall_means['not_formed_out_degree_first_hop'])))
#
#         info_file.write("Out-degree Second Hop:\n\tFEM:{0:.3f} \t NFEM:{1:.3f}\n\n"
#                         .format(np.mean(overall_means['formed_out_degree_second_hop']),
#                                 np.mean(overall_means['not_formed_out_degree_second_hop'])))
#
#     print("Done with {0} triangle type".format(triangle_type))
#
# Parallel(n_jobs=9)(delayed(test_directed_triangle)(tri_type) for tri_type in triangle_types)
