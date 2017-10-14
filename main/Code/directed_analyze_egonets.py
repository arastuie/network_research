import os
import pickle
import numpy as np
import helpers as h
import networkx as nx
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import gplus_hop_degree_directed_analysis_cdf as analyzer


data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes'

Parallel(n_jobs=9)(delayed(analyzer.gplus_run_hop_degree_directed_analysis)(ego_net_file)
                   for ego_net_file in os.listdir(data_file_base_path))
