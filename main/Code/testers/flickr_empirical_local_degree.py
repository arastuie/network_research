import pickle
import numpy as np
import networkx as nx
import directed_graphs_helpers as dh
from functools import partial
import flickr_helpers as flickr
from multiprocessing import Pool, Manager


def local_degree_empirical_analysis(ego_node, orig_snapshots, all_results, n_nodes, counter):
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

    ego_net_snapshots = []

    for s in range(len(orig_snapshots)):
        ego_net_snapshots.append(nx.ego_graph(orig_snapshots[s], ego_node, radius=2, center=True, undirected=True))

        if nx.number_of_nodes(ego_net_snapshots[0]) > 100000:
            counter.value += 1
            print("Progress: {0:2.2f}%".format(100 * counter.value / n_nodes), end='\r')
            return

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

        all_results[triangle_type]['Global']['id-formed'].append(np.mean(global_snapshots_formed_z_in_degree))
        all_results[triangle_type]['Global']['id-not-formed'].append(np.mean(global_snapshots_not_formed_z_in_degree))
        all_results[triangle_type]['Global']['od-formed'].append(np.mean(global_snapshots_formed_z_out_degree))
        all_results[triangle_type]['Global']['od-not-formed'].append(np.mean(global_snapshots_not_formed_z_out_degree))

        all_results[triangle_type]['Local']['id-formed'].append(np.mean(local_snapshots_formed_z_in_degree))
        all_results[triangle_type]['Local']['id-not-formed'].append(np.mean(local_snapshots_not_formed_z_in_degree))
        all_results[triangle_type]['Local']['od-formed'].append(np.mean(local_snapshots_formed_z_out_degree))
        all_results[triangle_type]['Local']['od-not-formed'].append(np.mean(local_snapshots_not_formed_z_out_degree))

    counter.value += 1
    print("Progress: {0:2.2f}%".format(100 * counter.value / n_nodes), end='\r')
    return


if __name__ == '__main__':

    # # Here is the format of the all_results for every triangle type
    # all_results = {
    #     'T01': {
    #         'Global': {
    #             'id-formed': [],
    #             'id-not-formed': [],
    #             'od-formed': [],
    #             'od-not-formed': []
    #         },
    #         'Local': {
    #             'id-formed': [],
    #             'id-not-formed': [],
    #             'od-formed': [],
    #             'od-not-formed': []
    #         }
    #     }
    # }

    results_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/flickr/pickle-files/'
    triangle_types_list = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
    edge_types = ['id-formed', 'id-not-formed', 'od-formed', 'od-not-formed']

    # Read in the pickle file of the snapshot flickr graph
    orig_snaps = flickr.read_snapshot_pickle_file()

    n_nodes_to_analyze = 50
    first_snap_nodes = list(orig_snaps[0].nodes())
    nodes_to_analyze = list(np.random.choice(first_snap_nodes, n_nodes_to_analyze))

    print("Number of nodes to be analyzed: {0}".format(n_nodes_to_analyze))

    manager = Manager()
    cnt = manager.Value('i', 0)
    all_results = manager.dict()
    for triangle_type in triangle_types_list:
        all_results[triangle_type] = manager.dict()
        all_results[triangle_type]['Global'] = manager.dict()
        all_results[triangle_type]['Local'] = manager.dict()

        for et in edge_types:
            all_results[triangle_type]['Global'][et] = manager.list()
            all_results[triangle_type]['Local'][et] = manager.list()

    local_degree_empirical_analysis_partial = partial(local_degree_empirical_analysis, orig_snapshots=orig_snaps,
                                                      all_results=all_results, n_nodes=n_nodes_to_analyze, counter=cnt)

    print("Start parallel analyses...\n")
    with Pool() as p:
        p.map(local_degree_empirical_analysis_partial, nodes_to_analyze)

    p.join()
    # end of analysis

    # convert all the shared memory to regular variables
    regular_all_results = {}
    for triangle_type in triangle_types_list:
        regular_all_results[triangle_type] = {}
        regular_all_results[triangle_type]['Global'] = {}
        regular_all_results[triangle_type]['Local'] = {}

        for et in edge_types:
            regular_all_results[triangle_type]['Global'][et] = list(all_results[triangle_type]['Global'][et])
            regular_all_results[triangle_type]['Local'][et] = list(all_results[triangle_type]['Local'][et])

    # write the results to hard drive
    with open('{0}/total-result.pckl'.format(results_path), 'wb') as f:
        pickle.dump(regular_all_results, f, protocol=-1)

    # write the results to hard drive
    with open('{0}/nodes_analyzed_in_total_results.pckl'.format(results_path), 'wb') as f:
        pickle.dump(nodes_to_analyze, f, protocol=-1)

    # Read in the results
    with open('{0}/total-result.pckl'.format(results_path), 'rb') as f:
        res = pickle.load(f)

    print("\nResults:")
    for triangle_type in triangle_types_list:
        print(triangle_type)
        for et in edge_types:
            print("Global", et, regular_all_results[triangle_type]['Global'][et])
            print("Local", et, regular_all_results[triangle_type]['Local'][et])
