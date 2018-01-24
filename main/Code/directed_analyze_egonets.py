import os
import pickle
import numpy as np
import helpers as h
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import link_prediction_helpers as dh
import gplus_hop_degree_directed_analysis_cdf as analyzer


data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes'
triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]

temp_skipped_files = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/combined/temp'

Parallel(n_jobs=20)(delayed(analyzer.gplus_run_hop_degree_directed_analysis)(ego_net_file)
                    for ego_net_file in os.listdir(data_file_base_path))


# Parallel(n_jobs=15)(delayed(dh.run_link_prediction_comparison_on_directed_graph_all_types)(ego_net_file, top_k_values)
#                     for ego_net_file in os.listdir(data_file_base_path))

# Parallel(n_jobs=16)(delayed(dh.run_link_prediction_comparison_on_directed_graph_all_types_based_on_empirical)
#                     (ego_net_file, top_k_values) for ego_net_file in os.listdir(data_file_base_path))

# Parallel(n_jobs=10)(delayed(dh.run_link_prediction_comparison_on_directed_graph_combined_types)
#                    (ego_net_file, top_k_values) for ego_net_file in os.listdir(temp_skipped_files))

# Parallel(n_jobs=16)(delayed(dh.run_link_prediction_comparison_on_directed_graph_combined_types)
#                    (ego_net_file, top_k_values) for ego_net_file in os.listdir(data_file_base_path))
