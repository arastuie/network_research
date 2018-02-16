import os
import sys
import pickle
import numpy as np
import networkx as nx
import directed_graphs_helpers as dh


def gplus_run_hop_degree_directed_analysis(ego_net_file):
    data_file_base_path = '/shared/DataSets/GooglePlus_Gong2012/egocentric/egonet-files/first-hop-nodes/'
    result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/gplus/pickle-files/' \
                            'test-2-no-least-num-nodes/'

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    # return if the egonet is on the currently being analyzed list
    if os.path.isfile(result_file_base_path + 'temp-analyses-start/' + ego_net_file):
        return

    with open(result_file_base_path + 'temp-analyses-start/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    triangle_type_func = {
        'T01': dh.get_t01_type_nodes,
        'T02': dh.get_t02_type_nodes,
        'T03': dh.get_t03_type_nodes,
        'T04': dh.get_t04_type_nodes,
        'T05': dh.get_t05_type_nodes,
        'T06': dh.get_t06_type_nodes,
        'T07': dh.get_t07_type_nodes,
        'T08': dh.get_t08_type_nodes,
        'T09': dh.get_t09_type_nodes,
    }

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net = pickle.load(f)

    ego_net_snapshots = []

    # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
    if nx.number_of_nodes(ego_net) > 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)

        return

    for r in range(0, 4):
        temp_net = nx.DiGraph([(u, v, d) for u, v, d in ego_net.edges(data=True) if d['snapshot'] <= r])
        ego_net_snapshots.append(nx.ego_graph(temp_net, ego_node, radius=2, center=True, undirected=True))

    for triangle_type in triangle_type_func.keys():
        local_snapshots_formed_z_in_degree = []
        local_snapshots_formed_z_out_degree = []
        local_snapshots_not_formed_z_in_degree = []
        local_snapshots_not_formed_z_out_degree = []

        global_snapshots_formed_z_in_degree = []
        global_snapshots_formed_z_out_degree = []
        global_snapshots_not_formed_z_in_degree = []
        global_snapshots_not_formed_z_out_degree = []

        # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
        for i in range(len(ego_net_snapshots) - 1):
            first_hop_nodes, second_hop_nodes, v_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i], ego_node)

            len_first_hop = len(first_hop_nodes)
            tot_num_nodes = nx.number_of_nodes(ego_net_snapshots[i])

            # if len_first_hop < 10:
            #     continue

            # Checks whether or not any edge were formed and not formed, if not skips to next snapshot
            has_any_formed = False
            has_any_not_formed = False
            for v in v_nodes:
                if ego_net_snapshots[i + 1].has_edge(ego_node, v):
                    has_any_formed = True
                else:
                    has_any_not_formed = True

                if has_any_formed and has_any_not_formed:
                    break

            if not has_any_formed or not has_any_not_formed:
                continue

            # ANALYSIS
            local_formed_z_in_degree = []
            local_formed_z_out_degree = []
            local_not_formed_z_in_degree = []
            local_not_formed_z_out_degree = []

            global_formed_z_in_degree = []
            global_formed_z_out_degree = []
            global_not_formed_z_in_degree = []
            global_not_formed_z_out_degree = []

            for v in v_nodes:
                local_temp_in_degree = []
                local_temp_out_degree = []

                global_temp_in_degree = []
                global_temp_out_degree = []

                for z in v_nodes[v]:
                    z_preds = set(ego_net_snapshots[i].predecessors(z))
                    z_succs = set(ego_net_snapshots[i].successors(z))

                    local_temp_in_degree.append(len(z_preds.intersection(first_hop_nodes)))
                    local_temp_out_degree.append(len(z_succs.intersection(first_hop_nodes)))

                    global_temp_in_degree.append(len(z_preds))
                    global_temp_out_degree.append(len(z_succs))

                if ego_net_snapshots[i + 1].has_edge(ego_node, v):
                    local_formed_z_in_degree.append(np.mean(local_temp_in_degree))
                    local_formed_z_out_degree.append(np.mean(local_temp_out_degree))

                    global_formed_z_in_degree.append(np.mean(global_temp_in_degree))
                    global_formed_z_out_degree.append(np.mean(global_temp_out_degree))
                else:
                    local_not_formed_z_in_degree.append(np.mean(local_temp_in_degree))
                    local_not_formed_z_out_degree.append(np.mean(local_temp_out_degree))

                    global_not_formed_z_in_degree.append(np.mean(global_temp_in_degree))
                    global_not_formed_z_out_degree.append(np.mean(global_temp_out_degree))

            # normalizing by the number of nodes in the first hop
            local_snapshots_formed_z_in_degree.append(np.mean(local_formed_z_in_degree) / len_first_hop)
            local_snapshots_formed_z_out_degree.append(np.mean(local_formed_z_out_degree) / len_first_hop)
            local_snapshots_not_formed_z_in_degree.append(np.mean(local_not_formed_z_in_degree) / len_first_hop)
            local_snapshots_not_formed_z_out_degree.append(np.mean(local_not_formed_z_out_degree) / len_first_hop)

            # normalizing by the number of nodes in the entire snapshot
            global_snapshots_formed_z_in_degree.append(np.mean(global_formed_z_in_degree) / tot_num_nodes)
            global_snapshots_formed_z_out_degree.append(np.mean(global_formed_z_out_degree) / tot_num_nodes)
            global_snapshots_not_formed_z_in_degree.append(np.mean(global_not_formed_z_in_degree) / tot_num_nodes)
            global_snapshots_not_formed_z_out_degree.append(np.mean(global_not_formed_z_out_degree) / tot_num_nodes)

        # Return if there was no V node found
        if len(local_snapshots_formed_z_in_degree) == 0:
            continue

        with open(result_file_base_path + triangle_type + '/' + ego_net_file, 'wb') as f:
            pickle.dump([np.mean(local_snapshots_formed_z_in_degree),
                         np.mean(global_snapshots_formed_z_in_degree),
                         np.mean(local_snapshots_formed_z_out_degree),
                         np.mean(global_snapshots_formed_z_out_degree),
                         np.mean(local_snapshots_not_formed_z_in_degree),
                         np.mean(global_snapshots_not_formed_z_in_degree),
                         np.mean(local_snapshots_not_formed_z_out_degree),
                         np.mean(global_snapshots_not_formed_z_out_degree)], f, protocol=-1)

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    print("Analyzed ego net {0}".format(ego_net_file))
