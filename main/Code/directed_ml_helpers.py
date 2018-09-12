import os
import glob
import math
import pickle
import numpy as np
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import directed_graphs_helpers as dgh

#    0           1       2   3   4    5        6           7       8       9                  10                 11                   12               13
# egonet-id, v-node-id, CN, AA, CAR, CCLP, LD-undirectd, LD-in, LD-out, snapshot_index, #_nodes_first_hop, #_nodes_second_hop, #_of_edges_in_egonet, formed?

def evaluate_lp_measures(ego_net_file, data_file_base_path, result_file_base_path, skip_over_100k=True):
    # egonet-id, v-node-id, CN, AA, CAR, CCLP, LD-undirectd, LD-in, LD-out, snapshot_index, #_nodes_first_hop,
    # #_nodes_second_hop, #_of_edges_in_egonet, formed?
    start_time = datetime.now()

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if skip_over_100k and os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    num_nodes = nx.number_of_nodes(ego_net_snapshots[-1])
    # if the number of nodes in the last snapshot of the network is big, skip it and save a file in skipped-nets
    if skip_over_100k and num_nodes >= 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)
        return

    gathered_data = 0
    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes, _, v_nodes = dgh.get_combined_type_nodes(ego_net_snapshots[i], ego_node)

        v_nodes_list = list(v_nodes.keys())

        snapshot_data = np.zeros((len(v_nodes_list), 14))
        snapshot_data[:, 0] = ego_node
        snapshot_data[:, 9] = i
        snapshot_data[:, 10] = len(first_hop_nodes)
        snapshot_data[:, 11] = len(v_nodes_list)
        snapshot_data[:, 12] = ego_net_snapshots[i].number_of_edges()

        for v_i in range(0, len(v_nodes_list)):
            snapshot_data[v_i, 1] = v_nodes_list[v_i]
            snapshot_data[v_i, -1] = 1 if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]) else 0

        # a dict of info on z nodes. Every key points to a list [Local_Degree, Local_indegre, local_outdegree, aa, cclp]
        z_info = {}
        # undirected network for CCLP and CAR
        undirected_ego_net = ego_net_snapshots[i].to_undirected()
        for z in first_hop_nodes:
            z_successors = set(ego_net_snapshots[i].successors(z))
            z_predecessors = set(ego_net_snapshots[i].predecessors(z))
            z_neighbors = z_successors.union(z_predecessors)

            # This should be the intersection of z_neighbors with the union of nodes in first and second hops
            z_global_degree = len(z_neighbors)

            z_local_in_degree = len(z_predecessors.intersection(first_hop_nodes))
            z_local_out_degree = len(z_successors.intersection(first_hop_nodes))
            z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

            y = z_global_degree - z_local_degree

            # if y = 1, then the z node has no neighbor in the second hop, thus no need to compute
            if y == 1:
                continue

            z_info[z] = []
            # ld
            z_info[z].append(math.log(z_local_degree + 1))
            # lid
            z_info[z].append(math.log(z_local_in_degree + 1))
            # lod
            z_info[z].append(math.log(z_local_out_degree + 1))
            # aa
            z_info[z].append(1 / math.log(z_global_degree))
            # cclp
            z_info[z].append(nx.triangles(undirected_ego_net, z) / (z_global_degree * (z_global_degree - 1) / 2))

        for v_i in range(len(v_nodes_list)):
            num_cn = len(v_nodes[v_nodes_list[v_i]])

            # cn score
            snapshot_data[v_i, 2] = num_cn

            # car score
            lcl = 0
            if num_cn > 1:
                for zz_i in range(num_cn - 1):
                    for zzz_i in range(zz_i + 1, num_cn):
                        v_z_list = v_nodes[v_nodes_list[v_i]]
                        if ego_net_snapshots[i].has_edge(v_z_list[zz_i], v_z_list[zzz_i]) or \
                                ego_net_snapshots[i].has_edge(v_z_list[zzz_i], v_z_list[zz_i]):
                            lcl += 1

            snapshot_data[v_i, 4] = num_cn * lcl

            temp_ld = 0
            temp_lid = 0
            temp_lod = 0
            temp_aa = 0
            temp_cclp = 0

            for z in v_nodes[v_nodes_list[v_i]]:
                temp_ld += z_info[z][0]
                temp_lid += z_info[z][1]
                temp_lod += z_info[z][2]
                temp_aa += z_info[z][3]
                temp_cclp += z_info[z][4]

            snapshot_data[v_i, 3] = temp_aa
            snapshot_data[v_i, 5] = temp_cclp
            snapshot_data[v_i, 6] = temp_ld
            snapshot_data[v_i, 7] = temp_lid
            snapshot_data[v_i, 8] = temp_aa

        if type(gathered_data) == int:
            gathered_data = snapshot_data
        else:
            gathered_data = np.concatenate((gathered_data, snapshot_data), axis=0)


    # Saving the entire data in a csv
    print("fix this")
    # np.savetxt("{}results/{}.txt".format(result_file_base_path, ego_node), gathered_data, delimiter=",")

    print("Analyzed ego net: {0} - Duration: {1} - Num nodes: {2} - Formed: {3}"
          .format(ego_net_file, datetime.now() - start_time, num_nodes, np.sum(gathered_data[:, -1])))

    # save an empty file in analyzed_egonets to know which ones were analyzed
    # with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
    #     pickle.dump(0, f, protocol=-1)

    return


def combine_all_egonet_data(result_base_path, num_rows=0):
    combined_result_path = result_base_path + 'combined/'
    result_base_path = result_base_path + 'results/'
    num_egonets = len(os.listdir(result_base_path))

    # count number of rows if not passed
    cnt = 0
    if num_rows == 0:
        print("Count the total number of rows.")
        for egonet_data_file_name in os.listdir(result_base_path):
            egonet_data = np.load(result_base_path + egonet_data_file_name)
            num_rows += np.shape(egonet_data)[0]
            print("{:3.2f}%".format(100 * cnt / num_egonets), end='\r')
            cnt += 1
        print()

    # combine all sets
    cnt = 0
    nxt_row_start_index = 0
    dataset = np.zeros((num_rows, 14))
    print("Start combining the egonet files...")
    for egonet_data_file_name in os.listdir(result_base_path):
        egonet_data = np.load(result_base_path + egonet_data_file_name)

        data_length = np.shape(egonet_data)[0]

        dataset[nxt_row_start_index:nxt_row_start_index + data_length, :] = egonet_data
        nxt_row_start_index = nxt_row_start_index + data_length

        print("{:3.2f}%".format(100 * cnt / num_egonets), end='\r')
        cnt += 1

    print()
    if not os.path.exists(combined_result_path):
        os.makedirs(combined_result_path)

    np.save(combined_result_path + "all.npy", dataset)
    print("Done.")
    return


def split_data_based_on_snapshot(result_base_path):
    # this method splits the data into snapshot 1, snapshot 2, and rest of them
    combined_result_path = result_base_path + 'combined/'
    if not os.path.exists(combined_result_path + 'all.npy'):
        print("Combined results not found!")
        return

    print("Loading the dataset...")
    dataset = np.load(combined_result_path + 'all.npy')
    print("Dataset loaded.")

    # snapshot 0
    split_dataset = dataset[np.where(dataset[:, 9] == 0)[0], :]
    np.save("{}snapshot-{}-all.npy".format(combined_result_path, 0), split_dataset)
    del split_dataset
    print("Snapshot 0 done.")

    # snapshot 1
    split_dataset = dataset[np.where(dataset[:, 9] == 1)[0], :]
    np.save("{}snapshot-{}-all.npy".format(combined_result_path, 1), split_dataset)
    del split_dataset
    print("Snapshot 1 done.")

    # snapshots > 1
    split_dataset = dataset[np.where(dataset[:, 9] > 1)[0], :]
    np.save("{}snapshots-after-1-all.npy".format(combined_result_path), split_dataset)
    del split_dataset
    print("Snapshots after 1 done.")

    return

# def testt1(result_base_path):
#     result_base_path = result_base_path + 'results/'
#     files_list = list(glob.glob(result_base_path + "*.txt"))
#     print(len(files_list))
#     Parallel(n_jobs=10)(delayed(testt)(egonet_data_file_path)
#                         for egonet_data_file_path in files_list)
#
#
# def testt(egonet_data_file_path):
#     try:
#         with open(egonet_data_file_path, 'rb') as f:
#             egonet_data = np.loadtxt(f, delimiter=',')
#     except ValueError:
#         os.remove(egonet_data_file_path)
#         return
#
#     if len(np.shape(egonet_data)) == 1:
#         egonet_data = np.reshape(egonet_data, (1, 14))
#
#     np.save(egonet_data_file_path[:-4] + ".npy", egonet_data)
#     os.remove(egonet_data_file_path)
#
#     return