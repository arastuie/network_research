import pickle
import helpers as h
import degree_based_analysis as a1
import num_cluster_per_cn_analysis as a2
import tot_num_cluster_of_all_cn_analysis as a3
import tot_num_cluster_based_num_cn_analysis as a4


# reading FB graph
# original_graph = h.read_facebook_graph()

# extracting ego centric networks
# ego_centric_networks, ego_nodes = h.get_ego_centric_networks_in_fb(original_graph, n=50, hop=2, center=True)

print("Loading Facebook largest 50 ego centric networks...")

with open('../Data/first_50_ego.pckl', 'rb') as f:
    ego_centric_networks, ego_nodes = pickle.load(f)

print("Analysing ego centric networks...")

num_cluster = 4

for o in range(len(ego_centric_networks)):

    a1.run_degree_based_analysis(ego_centric_networks[o], ego_nodes[o], o, True, '../Plots/degree_based')

    a2.run_num_cluster_per_cn_analysis(ego_centric_networks[o], ego_nodes[o], num_cluster, o, True,
                                       '../Plots/cluster_per_node')

    a3.run_tot_num_cluster_of_all_cn_analysis(ego_centric_networks[o], ego_nodes[o], num_cluster, o, True,
                                              '../Plots/total_cluster')

    a4.run_tot_num_cluster_based_num_cn_analysis(ego_centric_networks[o], ego_nodes[o], num_cluster, o, True,
                                                 '../Plots/total_cluster_overall')


