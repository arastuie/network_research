import os
import pickle
import numpy as np
import networkx as nx
from functools import partial
import digg_net_helpers as digg
import link_prediction_helpers as lp_helpers
from multiprocessing import Pool, Manager


def run_lp_scores_on_an_ego(ego_and_index, original_snaps, scores_res, top_k_vals, list_of_scores, n_nodes, counter):
    # extract all ego nets for ease of computation
    ego = ego_and_index[0]
    snap_i = ego_and_index[1]

    ego_snaps = []
    for i in range(snap_i, len(original_snaps)):
        ego_snaps.append(nx.ego_graph(original_snaps[i], ego, radius=2, center=True))

    test_scores = lp_helpers.run_link_prediction_parallel(ego_snaps, ego, top_k_vals,
                                                          range(len(ego_snaps) - 1), list_of_scores)

    for k in top_k_vals:
        if len(test_scores[list_of_scores[0]][k]) > 0:
            for s_type in list_of_scores:
                scores_res[s_type][k].append(np.mean(test_scores[s_type][k]))

    counter.value += 1
    print("Progress: {0:2.2f}%".format(100 * counter.value / n_nodes), end='\r')


if __name__ == '__main__':
    results_path = '/shared/Results/EgocentricLinkPrediction/main/lp/digg/pickle_files/'
    # must be in sorted order
    top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]
    scores_list = ['cn', 'aa', 'dccn', 'dcaa', 'car', 'cclp']

    snapshot_duration_in_days = 90
    orig_snaps = digg.divide_to_snapshots(digg.read_graph(), snapshot_duration_in_days)

    # getting a list of ego nodes to be analyzed, (ego noe, the first snapshot it appeared in)
    ego_nodes_for_lp = []
    added_nodes = set()

    for snap_index in range(len(orig_snaps) - 1):
        nodes_in_snap = set(orig_snaps[snap_index].nodes())
        for node in nodes_in_snap:
            if node not in added_nodes:
                ego_nodes_for_lp.append((node, snap_index))
                added_nodes.add(node)

    n_nodes_to_analyze = len(ego_nodes_for_lp)
    print("Number of nodes to be analyzed: {0}".format(n_nodes_to_analyze))

    print("Start parallel analyses...\n")
    manager = Manager()
    cnt = manager.Value('i', 0)
    percent_scores = manager.dict()
    for score_type in scores_list:
        percent_scores[score_type] = manager.dict()

        for k in top_k_values:
            percent_scores[score_type][k] = manager.list()

    run_lp_scores_on_an_ego_partial = partial(run_lp_scores_on_an_ego, original_snaps=orig_snaps,
                                              scores_res=percent_scores, top_k_vals=top_k_values,
                                              list_of_scores=scores_list, n_nodes=n_nodes_to_analyze, counter=cnt)
    with Pool() as p:
        p.map(run_lp_scores_on_an_ego_partial, ego_nodes_for_lp)

    p.join()
    # end of analysis

    # convert all the shared memory to regular variables
    regular_percent_score = {}
    for score_type in scores_list:
        regular_percent_score[score_type] = {}

        for k in top_k_values:
            regular_percent_score[score_type][k] = list(percent_scores[score_type][k])

    # write the results to hard drive
    with open('{0}/total-result.pckl'.format(results_path), 'wb') as f:
        pickle.dump(regular_percent_score, f, protocol=-1)

    # Read in the results
    with open('{0}/total-result.pckl'.format(results_path), 'rb') as f:
        percent_tests = pickle.load(f)

    print("\nResults:")
    for score_type in scores_list:
        print(score_type)
        for k in top_k_values:
            print("\tTop {0} K -> {1}".format(k, np.mean(percent_tests[score_type][k])))


