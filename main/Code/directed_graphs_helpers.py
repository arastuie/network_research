import os
import math
import pickle
import numpy as np
import helpers as h
import networkx as nx
import warnings
from datetime import datetime
import matplotlib.pyplot as plt


def plot_formed_vs_not(formed, not_formed, xlabel, subtitle, overall_mean_formed, overall_mean_not_formed,
                       save_plot=False, save_path=''):
    fig = None
    if save_plot:
        fig = plt.figure(figsize=(15, 8), dpi=100)
    else:
        fig = plt.figure()

    n_row = math.ceil(len(formed) / 5)
    n_col = math.ceil(len(not_formed) / n_row)

    overall_means_formed = []
    overall_means_not_formed = []

    for i in range(len(formed)):
        formed_mean = np.mean(formed[i])
        not_formed_mean = np.mean(not_formed[i])

        overall_means_formed.append(formed_mean)
        overall_means_not_formed.append(not_formed_mean)
        p = fig.add_subplot(n_row, n_col, i + 1)

        p.hist(formed[i], color='r', alpha=0.8, weights=np.zeros_like(formed[i]) + 1. / len(formed[i]),
               label="FEM: {0:.2f}".format(formed_mean))

        p.hist(not_formed[i], color='b', alpha=0.5, weights=np.zeros_like(not_formed[i]) + 1. / len(not_formed[i]),
               label="NFEM: {0:.2f}".format(not_formed_mean))

        p.legend(loc='upper right')
        plt.ylabel('Relative Frequency')

        plt.xlabel(xlabel)
        plt.suptitle(subtitle)

    overall_mean_formed.append(np.mean(overall_means_formed))
    overall_mean_not_formed.append(np.mean(overall_means_not_formed))

    if not save_plot:
        plt.show()
    else:
        current_fig = plt.gcf()
        current_fig.savefig(save_path)

    plt.close(fig)


########## Local degree empirical analysis ##############
def get_t01_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t02_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)) - set(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t03_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)) - set(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t04_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)) - set(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)) - set(ego_net.predecessors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t05_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)).intersection(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t06_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node)) - set(ego_net.predecessors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t07_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.predecessors(ego_node)) - set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z)) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t08_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.predecessors(ego_node)) - set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_t09_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.predecessors(ego_node)) - set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)) - set(ego_net.predecessors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(v, ego_node):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def run_local_degree_empirical_analysis(ego_net_file, results_base_path, egonet_file_base_path, log_degree, skip_snaps,
                                        normalize, skip_over_100k=True):
    # log_degee (boolean): take log of the degrees before the mean
    # skip_snaps (boolean): skip snapshots where there is no edges formed
    # normalize (boolean): normalize the mean of each snapshots
    
    # return if the egonet is on the analyzed list
    if os.path.isfile(results_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if skip_over_100k and os.path.isfile(results_base_path + 'skipped_egonets/' + ego_net_file):
        return

    # return if the egonet is on the currently being analyzed list
    if os.path.isfile(results_base_path + 'temp-analyses-start/' + ego_net_file):
        return

    triangle_type_func = {
        'T01': get_t01_type_nodes,
        'T02': get_t02_type_nodes,
        'T03': get_t03_type_nodes,
        'T04': get_t04_type_nodes,
        'T05': get_t05_type_nodes,
        'T06': get_t06_type_nodes,
        'T07': get_t07_type_nodes,
        'T08': get_t08_type_nodes,
        'T09': get_t09_type_nodes,
    }

    with open(egonet_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
    if skip_over_100k and nx.number_of_nodes(ego_net_snapshots[0]) > 100000:
        with open(results_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)

        return

    with open(results_base_path + 'temp-analyses-start/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

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
            first_hop_nodes, second_hop_nodes, v_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i],
                                                                                           ego_node)

            # Checks whether or not any edge were formed and not formed, if not skips to next snapshot
            if skip_snaps:
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

            # dict of lists -> z: [local_in, local_out, global_in, global_out]
            z_degree_info = {}

            for z in first_hop_nodes:
                z_preds = set(ego_net_snapshots[i].predecessors(z))
                z_succs = set(ego_net_snapshots[i].successors(z))

                if log_degree:
                    z_degree_info[z] = np.log([len(z_preds.intersection(first_hop_nodes)) + 2,
                                               len(z_succs.intersection(first_hop_nodes)) + 2,
                                               len(z_preds) + 2,
                                               len(z_succs) + 2])
                else:
                    z_degree_info[z] = [len(z_preds.intersection(first_hop_nodes)),
                                        len(z_succs.intersection(first_hop_nodes)),
                                        len(z_preds),
                                        len(z_succs)]

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
                    local_temp_in_degree.append(z_degree_info[z][0])
                    local_temp_out_degree.append(z_degree_info[z][1])
                    global_temp_in_degree.append(z_degree_info[z][2])
                    global_temp_out_degree.append(z_degree_info[z][3])

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

            # if any of these arrays are empty due to no formed edges, the mean returns a nan value, which is what
            # we want. So ignore the warning.
            if normalize:
                # if any of these arrays are empty due to no formed edges, the mean returns a nan value, which is what
                # we want. So ignore the warning.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    len_first_hop = len(first_hop_nodes)
                    tot_num_nodes = nx.number_of_nodes(ego_net_snapshots[i])

                    # normalizing by the number of nodes in the first hop
                    local_snapshots_formed_z_in_degree.append(np.mean(local_formed_z_in_degree) / len_first_hop)
                    local_snapshots_formed_z_out_degree.append(np.mean(local_formed_z_out_degree) / len_first_hop)
                    local_snapshots_not_formed_z_in_degree.append(np.mean(local_not_formed_z_in_degree) / len_first_hop)
                    local_snapshots_not_formed_z_out_degree.append(np.mean(local_not_formed_z_out_degree) /
                                                                   len_first_hop)

                    # normalizing by the number of nodes in the entire snapshot
                    global_snapshots_formed_z_in_degree.append(np.mean(global_formed_z_in_degree) / tot_num_nodes)
                    global_snapshots_formed_z_out_degree.append(np.mean(global_formed_z_out_degree) / tot_num_nodes)
                    global_snapshots_not_formed_z_in_degree.append(np.mean(global_not_formed_z_in_degree) /
                                                                   tot_num_nodes)
                    global_snapshots_not_formed_z_out_degree.append(np.mean(global_not_formed_z_out_degree) /
                                                                    tot_num_nodes)

            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # normalizing by the number of nodes in the first hop
                    local_snapshots_formed_z_in_degree.append(np.mean(local_formed_z_in_degree))
                    local_snapshots_formed_z_out_degree.append(np.mean(local_formed_z_out_degree))
                    local_snapshots_not_formed_z_in_degree.append(np.mean(local_not_formed_z_in_degree))
                    local_snapshots_not_formed_z_out_degree.append(np.mean(local_not_formed_z_out_degree))

                    # normalizing by the number of nodes in the entire snapshot
                    global_snapshots_formed_z_in_degree.append(np.mean(global_formed_z_in_degree))
                    global_snapshots_formed_z_out_degree.append(np.mean(global_formed_z_out_degree))
                    global_snapshots_not_formed_z_in_degree.append(np.mean(global_not_formed_z_in_degree))
                    global_snapshots_not_formed_z_out_degree.append(np.mean(global_not_formed_z_out_degree))

        # Return if there was no V node found
        if len(local_snapshots_formed_z_in_degree) == 0:
            continue

        with open(results_base_path + triangle_type + '/' + ego_net_file, 'wb') as f:
            # if any of these arrays are all nans due to no formed edges in all snapshots, the nanmean returns a nan
            # value, which is what we want. So ignore the warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pickle.dump([np.nanmean(local_snapshots_formed_z_in_degree),
                             np.nanmean(global_snapshots_formed_z_in_degree),
                             np.nanmean(local_snapshots_formed_z_out_degree),
                             np.nanmean(global_snapshots_formed_z_out_degree),
                             np.nanmean(local_snapshots_not_formed_z_in_degree),
                             np.nanmean(global_snapshots_not_formed_z_in_degree),
                             np.nanmean(local_snapshots_not_formed_z_out_degree),
                             np.nanmean(global_snapshots_not_formed_z_out_degree)], f, protocol=-1)

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(results_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    # remove temp analyze file
    os.remove(results_base_path + 'temp-analyses-start/' + ego_net_file)

    print("Analyzed ego net {0}".format(ego_net_file))


def get_all_empirical_resutls(result_file_base_path, gather_individual_results):
    # Results pickle files are in the following order
    #   local-formed-in-degree, global-formed-in-degree, local-formed-out-degree, global-formed-out-degree
    #   local-not-formed-in-degree, global-not-formed-in-degree, local-not-formed-out-degree,
    #   global-not-formed-out-degree

    triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
    gl_labels = ['Local', 'Global']
    z = 1.96

    if gather_individual_results:
        all_results = {
            'Global': {
                'id-formed': [],
                'id-not-formed': [],
                'od-formed': [],
                'od-not-formed': [],
                'id-formed-err': [],
                'id-not-formed-err': [],
                'od-formed-err': [],
                'od-not-formed-err': []
            },
            'Local': {
                'id-formed': [],
                'id-not-formed': [],
                'od-formed': [],
                'od-not-formed': [],
                'id-formed-err': [],
                'id-not-formed-err': [],
                'od-formed-err': [],
                'od-not-formed-err': []
            }
        }

        # grabbing all scores
        for t in range(len(triangle_types)):
            results = []

            # loading result data
            for result_file in os.listdir(result_file_base_path + triangle_types[t]):
                with open(result_file_base_path + triangle_types[t] + '/' + result_file, 'rb') as f:
                    egonet_result = pickle.load(f)

                results.append(egonet_result)
            results = np.array(results)

            for i in range(2):
                # computing mean
                all_results[gl_labels[i]]['id-formed'].append(np.nanmean(results[:, i]))
                all_results[gl_labels[i]]['id-not-formed'].append(np.nanmean(results[:, i + 4]))
                all_results[gl_labels[i]]['od-formed'].append(np.nanmean(results[:, i + 2]))
                all_results[gl_labels[i]]['od-not-formed'].append(np.nanmean(results[:, i + 6]))

                # computing 95% confidence interval
                all_results[gl_labels[i]]['id-formed-err'].append(h.get_mean_ci(results[:, i], z, has_nan=True))
                all_results[gl_labels[i]]['id-not-formed-err'].append(h.get_mean_ci(results[:, i + 4], z, has_nan=True))
                all_results[gl_labels[i]]['od-formed-err'].append(h.get_mean_ci(results[:, i + 2], z, has_nan=True))
                all_results[gl_labels[i]]['od-not-formed-err'].append(h.get_mean_ci(results[:, i + 6], z, has_nan=True))

            print(triangle_types[t] + ": Done")

        # Create directory if not exists
        if not os.path.exists(result_file_base_path + 'all-scores'):
            os.makedirs(result_file_base_path + 'all-scores')

        with open(result_file_base_path + 'all-scores/all-types-plot.pckle', 'wb') as f:
            pickle.dump(all_results, f, protocol=-1)

    else:
        with open(result_file_base_path + 'all-scores/all-types-plot.pckle', 'rb') as f:
            all_results = pickle.load(f)

    return  all_results


def plot_local_degree_empirical_results(result_file_base_path, plot_save_path, gather_individual_results=False):
    # Results pickle files are in the following order
    #   local-formed-in-degree, global-formed-in-degree, local-formed-out-degree, global-formed-out-degree
    #   local-not-formed-in-degree, global-not-formed-in-degree, local-not-formed-out-degree,
    #   global-not-formed-out-degree

    triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
    gl_labels = ['Local', 'Global']

    all_results = get_all_empirical_resutls(result_file_base_path, gather_individual_results)

    # plotting
    bar_width = 0.20
    opacity = 0.6
    error_config = {'ecolor': '0.3', 'capsize': 1, 'lw': 1, 'capthick': 1}
    bar_legends = ['In-degree Formed', 'In-degree Not Formed', 'Out-degree Formed', 'Out-degree Not Formed']
    dif_results = ['id-formed', 'id-not-formed', 'od-formed', 'od-not-formed']
    bar_color = ['r', 'b', 'g', 'y']

    for i_degree in range(2):
        plt.rc('legend', fontsize=14)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=14)
        fig, ax = plt.subplots()

        for i_bar in range(len(dif_results)):
            plt.bar(np.arange(len(triangle_types)) + bar_width * i_bar,
                    all_results[gl_labels[i_degree]][dif_results[i_bar]],
                    bar_width,
                    alpha=opacity,
                    color=bar_color[i_bar],
                    yerr=all_results[gl_labels[i_degree]][dif_results[i_bar] + '-err'],
                    error_kw=error_config,
                    label=bar_legends[i_bar])

        plt.xlabel('Triangle Types', fontsize=16)
        plt.ylabel('Mean Normalized {0} Degree'.format(gl_labels[i_degree]), fontsize=16)
        plt.xticks(np.arange(len(triangle_types)) + bar_width * 1.5, triangle_types)

        plt.legend(loc='upper left')
        # if i_degree == 0:
        #     plt.ylim(ymax=.51)
        plt.tight_layout()
        plt.savefig('{0}/overall-{1}.pdf'.format(plot_save_path, gl_labels[i_degree]), format='pdf')
        plt.clf()


def plot_local_degree_empirical_cdf(result_file_base_path, plot_save_path, triangle_types='all',
                                    separete_in_out_degree=False, gather_individual_results=False):
    # Results pickle files are in the following order
    #   local-formed-in-degree, global-formed-in-degree, local-formed-out-degree, global-formed-out-degree
    #   local-not-formed-in-degree, global-not-formed-in-degree, local-not-formed-out-degree,
    #   global-not-formed-out-degree

    if triangle_types == 'all':
        triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']

    gl_labels = ['Local', 'Global']
    alpha = 0.05

    # grabbing all scores
    if gather_individual_results:
        for triangle_type in triangle_types:
            results = []

            # loading result data
            for result_file in os.listdir(result_file_base_path + triangle_type):
                with open(result_file_base_path + triangle_type + '/' + result_file, 'rb') as f:
                    egonet_result = pickle.load(f)

                results.append(egonet_result)
            results = np.array(results)

            lower_bonds = []
            upper_bonds = []
            for ii in range(8):
                lb, ub = h.get_ecdf_bands(results[:, ii], alpha)
                lower_bonds.append(lb)
                upper_bonds.append(ub)

            with open(result_file_base_path + 'all-scores/cdf-' + triangle_type + '-all.pckle', 'wb') as f:
                pickle.dump([results, lower_bonds, upper_bonds], f, protocol=-1)

            print(triangle_type + ": Done")

        print("Done!")

    # plotting
    if not separete_in_out_degree:
        for triangle_type in triangle_types:
            with open(result_file_base_path + 'all-scores/cdf-' + triangle_type + '-all.pckle', 'rb') as f:
                res, lbs, ubs = pickle.load(f)

            for i in range(0, 2):
                plt.rc('legend', fontsize=14.5)
                plt.rc('xtick', labelsize=15)
                plt.rc('ytick', labelsize=15)

                h.add_ecdf_with_band_plot(res[:, i], lbs[i], ubs[i], 'In-degree Formed', 'r')
                h.add_ecdf_with_band_plot(res[:, i + 4], lbs[i + 4], ubs[i + 4], 'In-degree Not Formed', 'b')

                h.add_ecdf_with_band_plot(res[:, i + 2], lbs[i + 2], ubs[i + 2], 'Out-degree Formed', 'y')
                h.add_ecdf_with_band_plot(res[:, i + 6], lbs[i + 6], ubs[i + 6], 'Out-degree Not Formed', 'g')

                plt.ylabel('Empirical CDF', fontsize=20)
                plt.xlabel('Mean Normalized {0} Degree'.format(gl_labels[i]), fontsize=20)
                plt.legend(loc='lower right')
                plt.tight_layout()
                current_fig = plt.gcf()
                current_fig.savefig('{0}cdf-{1}-{2}.pdf'.format(plot_save_path, triangle_type, gl_labels[i]),
                                    format='pdf')
                plt.clf()

            print(triangle_type + ": Done")

        print("Done!")

    else:
        for triangle_type in triangle_types:
            with open(result_file_base_path + 'all-scores/cdf-' + triangle_type + '-all.pckle', 'rb') as f:
                res, lbs, ubs = pickle.load(f)

            for i in range(0, 2):
                plt.rc('legend', fontsize=20)
                plt.rc('xtick', labelsize=15)
                plt.rc('ytick', labelsize=15)

                h.add_ecdf_with_band_plot(res[:, i], lbs[i], ubs[i], 'Formed', 'r')
                h.add_ecdf_with_band_plot(res[:, i + 4], lbs[i + 4], ubs[i + 4], 'Not Formed', 'b')

                plt.ylabel('Empirical CDF', fontsize=20)
                plt.xlabel('Mean Normalized {0} In-degree'.format(gl_labels[i]), fontsize=20)
                plt.legend(loc='lower right')
                plt.tight_layout()
                current_fig = plt.gcf()
                current_fig.savefig('{0}cdf-{1}-indegree-{2}.pdf'.format(plot_save_path, triangle_type,
                                                                         gl_labels[i]), format='pdf')
                plt.clf()

                plt.rc('legend', fontsize=20)
                plt.rc('xtick', labelsize=15)
                plt.rc('ytick', labelsize=15)

                h.add_ecdf_with_band_plot(res[:, i + 2], lbs[i + 2], ubs[i + 2], 'Formed', 'r')
                h.add_ecdf_with_band_plot(res[:, i + 6], lbs[i + 6], ubs[i + 6], 'Not Formed', 'b')

                plt.ylabel('Empirical CDF', fontsize=20)
                plt.xlabel('Mean Normalized {0} Out-degree'.format(gl_labels[i]), fontsize=20)
                plt.legend(loc='lower right')
                plt.tight_layout()
                current_fig = plt.gcf()
                current_fig.savefig('{0}cdf-{1}-outdegree-{2}.pdf'.format(plot_save_path, triangle_type,
                                                                          gl_labels[i]), format='pdf')
                plt.clf()

            print(triangle_type + ": Done")

        print("Done!")


def local_degree_empirical_result_comparison(result_file_base_path, include_conf_intervals,
                                             gather_individual_results=False):
    # Results pickle files are in the following order
    #   local-formed-in-degree, global-formed-in-degree, local-formed-out-degree, global-formed-out-degree
    #   local-not-formed-in-degree, global-not-formed-in-degree, local-not-formed-out-degree,
    #   global-not-formed-out-degree
    triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06']

    all_results = get_all_empirical_resutls(result_file_base_path, gather_individual_results)
    print()
    # The print out is optimized for excel
    print('Global')
    for t_type in triangle_types:
        print(t_type, end=',')
        print(t_type, end=',')
    print()
    for _ in triangle_types:
        print('In-degree', end=',')
        print('out-degree', end=',')
    print()

    in_degree_diff = []
    out_degree_diff = []

    for i in range(len(triangle_types)):
        in_degree_diff_temp = all_results['Global']['id-not-formed'][i] - all_results['Global']['id-formed'][i]
        out_degree_diff_temp = all_results['Global']['od-not-formed'][i] - all_results['Global']['od-formed'][i]

        if include_conf_intervals:
            in_degree_diff_temp -= all_results['Global']['id-not-formed-err'][i] + \
                                   all_results['Global']['id-formed-err'][i]
            out_degree_diff_temp -= all_results['Global']['od-not-formed-err'][i] + \
                                    all_results['Global']['od-formed-err'][i]

        in_degree_diff.append(in_degree_diff_temp / all_results['Global']['id-not-formed'][i])
        out_degree_diff.append(out_degree_diff_temp / all_results['Global']['od-not-formed'][i])

        print(in_degree_diff_temp, end=',')
        print(out_degree_diff_temp, end=',')

    print("\nAverage global in-degree difference:{}".format(np.mean(in_degree_diff)))
    print("Average global out-degree difference:{}".format(np.mean(out_degree_diff)))

    print()
    print('Local')
    for t_type in triangle_types:
        print(t_type, end=',')
        print(t_type, end=',')
    print()
    for _ in triangle_types:
        print('In-degree', end=',')
        print('out-degree', end=',')
    print()

    in_degree_diff = []
    out_degree_diff = []

    for i in range(len(triangle_types)):
        in_degree_diff_temp = all_results['Local']['id-formed'][i] - all_results['Local']['id-not-formed'][i]
        out_degree_diff_temp = all_results['Local']['od-formed'][i] - all_results['Local']['od-not-formed'][i]

        if include_conf_intervals:
            in_degree_diff_temp -= all_results['Local']['id-formed-err'][i] + \
                                   all_results['Local']['id-not-formed-err'][i]
            out_degree_diff_temp -= all_results['Local']['od-formed-err'][i] + \
                                    all_results['Local']['od-not-formed-err'][i]

        in_degree_diff.append(in_degree_diff_temp / all_results['Local']['id-formed'][i])
        out_degree_diff.append(out_degree_diff_temp / all_results['Local']['od-formed'][i])

        print(in_degree_diff_temp, end=',')
        print(out_degree_diff_temp, end=',')

    print("\nAverage local in-degree difference:{}".format(np.mean(in_degree_diff)))
    print("Average local out-degree difference:{}".format(np.mean(out_degree_diff)))


########## Links formed in triad ratio analysis ##############
def get_t01_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego.intersection(predecessors_of_the_ego)

    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = (second_hop_nodes - successors_of_the_ego)

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t02_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego.intersection(predecessors_of_the_ego)
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)) - set(ego_net.predecessors(z))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = second_hop_nodes - successors_of_the_ego

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t03_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego - predecessors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = second_hop_nodes - successors_of_the_ego

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t04_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego - predecessors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)) - set(ego_net.predecessors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = second_hop_nodes - successors_of_the_ego

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t05_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego.intersection(predecessors_of_the_ego)
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = second_hop_nodes - successors_of_the_ego

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t06_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = successors_of_the_ego - predecessors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = second_hop_nodes - successors_of_the_ego

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t07_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = predecessors_of_the_ego - successors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = set(ego_net.successors(z)).intersection(ego_net.predecessors(z))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = second_hop_nodes - successors_of_the_ego

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t08_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = predecessors_of_the_ego - successors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.predecessors(z)) - set(ego_net.successors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = second_hop_nodes - successors_of_the_ego

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def get_t09_type_second_hop_nodes(ego_net, ego_node):
    successors_of_the_ego = set(ego_net.successors(ego_node))
    predecessors_of_the_ego = set(ego_net.predecessors(ego_node))

    first_hop_nodes = predecessors_of_the_ego - successors_of_the_ego
    second_hop_nodes = set()

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)) - set(ego_net.predecessors(z)))
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

    # remove nodes in the second hop which are already in the first hop
    second_hop_nodes = second_hop_nodes - first_hop_nodes

    # # remove all nodes in the second hop that have any edge with the ego
    # second_hop_nodes = second_hop_nodes - (successors_of_the_ego.union(predecessors_of_the_ego))

    # remove all nodes in the second hop that ego already follows.
    second_hop_nodes = second_hop_nodes - successors_of_the_ego

    # only allow nodes in the second hop that follow the ego
    second_hop_nodes = second_hop_nodes.intersection(predecessors_of_the_ego)

    # remove the ego node from the second hop
    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    num_nodes = len(second_hop_nodes) + len(first_hop_nodes) + 1
    return second_hop_nodes, num_nodes


def empirical_triad_links_formed_ratio(ego_net_file, data_file_base_path, result_file_base_path, skip_over_100k=True):
    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if skip_over_100k and os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    # return if the egonet is on the currently being analyzed list
    if os.path.isfile(result_file_base_path + 'temp-analyses-start/' + ego_net_file):
        return

    with open(result_file_base_path + 'temp-analyses-start/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    triangle_type_func = {
        'T01': get_t01_type_second_hop_nodes,
        'T02': get_t02_type_second_hop_nodes,
        'T03': get_t03_type_second_hop_nodes,
        'T04': get_t04_type_second_hop_nodes,
        'T05': get_t05_type_second_hop_nodes,
        'T06': get_t06_type_second_hop_nodes,
        'T07': get_t07_type_second_hop_nodes,
        'T08': get_t08_type_second_hop_nodes,
        'T09': get_t09_type_second_hop_nodes,
    }

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    # if the number of nodes in the network is really big, skip them and save a file in skipped-nets
    if skip_over_100k and ego_net_snapshots[-1].number_of_nodes() > 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)

        # remove temp analyze file
        if os.path.isfile(result_file_base_path + 'temp-analyses-start/' + ego_net_file):
            os.remove(result_file_base_path + 'temp-analyses-start/' + ego_net_file)

        return

    results = {
        'T01': {}, 'T02': {}, 'T03': {}, 'T04': {}, 'T05': {}, 'T06': {}, 'T07': {}, 'T08': {}, 'T09': {}
    }

    total_num_edges_formed = 0

    for t_type in results.keys():
        results[t_type]['num_edges_formed'] = []
        results[t_type]['num_nodes'] = []
        results[t_type]['num_second_hop_nodes'] = []

    results['total_num_nodes'] = []

    for i in range(len(ego_net_snapshots) - 1):
        results['total_num_nodes'].append(ego_net_snapshots[i].number_of_nodes())

        for triangle_type in triangle_type_func.keys():
            second_hop_nodes, num_nodes = triangle_type_func[triangle_type](ego_net_snapshots[i], ego_node)

            # number of nodes in the second hop that ego started to follow in the next snapshot
            next_snapshot_successors_of_the_ego = set(ego_net_snapshots[i + 1].successors(ego_node))
            num_edges_formed = len(next_snapshot_successors_of_the_ego.intersection(second_hop_nodes))

            results[triangle_type]['num_edges_formed'].append(num_edges_formed)
            results[triangle_type]['num_nodes'].append(num_nodes)
            results[triangle_type]['num_second_hop_nodes'].append(len(second_hop_nodes))

            total_num_edges_formed += num_edges_formed

    if total_num_edges_formed > 0:
        with open(result_file_base_path + 'results/' + ego_net_file, 'wb') as f:
            pickle.dump(results, f, protocol=-1)

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

    # remove temp analyze file
    if os.path.isfile(result_file_base_path + 'temp-analyses-start/' + ego_net_file):
        os.remove(result_file_base_path + 'temp-analyses-start/' + ego_net_file)

    print("Analyzed ego net {0}".format(ego_net_file))


def empirical_triad_list_formed_ratio_results_plot(result_file_base_path, plot_save_path,
                                                   gather_individual_results=False):
    triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']

    if gather_individual_results:
        fraction_of_all_formed_edges = {}
        edge_probability = {}

        for t in triangle_types:
            fraction_of_all_formed_edges[t] = []
            edge_probability[t] = []

        for result_file in os.listdir(result_file_base_path + 'results'):
            try:
                with open(result_file_base_path + 'results/' + result_file, 'rb') as f:
                    egonet_results = pickle.load(f)
            except EOFError:
                os.remove(result_file_base_path + 'results/' + result_file)

            # Calculating fraction ratio
            temp_snapshot_total_edges_formed = []

            for t_type in triangle_types:
                temp_snapshot_total_edges_formed.append(egonet_results[t_type]['num_edges_formed'])

            temp_snapshot_total_edges_formed = np.sum(temp_snapshot_total_edges_formed, axis=0)

            for t_type in triangle_types:
                temp_fraction = []
                for i in range(len(egonet_results[t_type]['num_edges_formed'])):
                    if temp_snapshot_total_edges_formed[i] != 0:
                        temp_fraction.append(egonet_results[t_type]['num_edges_formed'][i] /
                                             temp_snapshot_total_edges_formed[i])
                    # else:
                    #     temp_fraction.append(0)

                if len(temp_fraction) > 0:
                    fraction_of_all_formed_edges[t_type].append(np.mean(temp_fraction))


            # Calculating edge probability
            for t_type in triangle_types:
                temp_prob = []
                for i in range(len(egonet_results[t_type]['num_edges_formed'])):
                    if egonet_results[t_type]['num_second_hop_nodes'][i] != 0:
                        temp_prob.append(egonet_results[t_type]['num_edges_formed'][i] /
                                         egonet_results[t_type]['num_second_hop_nodes'][i])
                    # else:
                    #     temp_prob.append(0)

                if len(temp_prob) > 0:
                    edge_probability[t_type].append(np.mean(temp_prob))

        # Create directory if not exists
        if not os.path.exists(result_file_base_path + "cumulated_results"):
            os.makedirs(result_file_base_path + "cumulated_results")

        # Write data into a single file for fraction of all edges
        with open(result_file_base_path + "cumulated_results/fraction_of_all_formed_edges.pckle", 'wb') as f:
            pickle.dump(fraction_of_all_formed_edges, f, protocol=-1)

        # Write data into a single file for edge probability
        with open(result_file_base_path + "cumulated_results/edge_probability.pckle", 'wb') as f:
            pickle.dump(edge_probability, f, protocol=-1)
    else:
        with open(result_file_base_path + "cumulated_results/fraction_of_all_formed_edges.pckle", 'rb') as f:
            fraction_of_all_formed_edges = pickle.load(f)

        with open(result_file_base_path + "cumulated_results/edge_probability.pckle", 'rb') as f:
            edge_probability = pickle.load(f)

    plot_fraction_results = []
    plot_fraction_results_err = []
    plot_edge_prob_results = []
    plot_edge_prob_results_err = []
    for t_type in triangle_types:
        if len(fraction_of_all_formed_edges[t_type]) > 0:
            plot_fraction_results.append(np.mean(fraction_of_all_formed_edges[t_type]))
            plot_fraction_results_err.append(np.std(fraction_of_all_formed_edges[t_type]) /
                                             np.sqrt(len(fraction_of_all_formed_edges[t_type])))
        else:
            plot_fraction_results.append(0)
            plot_fraction_results_err.append(0)

        if len(edge_probability[t_type]) > 0:
            plot_edge_prob_results.append(np.mean(edge_probability[t_type]))
            plot_edge_prob_results_err.append(np.std(edge_probability[t_type]) /
                                              np.sqrt(len(edge_probability[t_type])))
        else:
            plot_edge_prob_results.append(0)
            plot_edge_prob_results_err.append(0)

    # plotting the fraction of edges
    plt.figure()
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.errorbar(np.arange(1, len(triangle_types) + 1), plot_fraction_results, yerr=plot_fraction_results_err,
                 color='b', fmt='--o')
    plt.ylabel('Fraction of All Edges', fontsize=22)
    plt.xlabel('Triad Type', fontsize=22)
    plt.tight_layout()
    current_fig = plt.gcf()
    plt.xticks(np.arange(1, len(triangle_types) + 1), triangle_types)
    current_fig.savefig('{0}triad_fraction_of_formed_edges.pdf'.format(plot_save_path), format='pdf')
    plt.clf()

    # plotting the edge probability
    plt.figure()
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.errorbar(np.arange(1, len(triangle_types) + 1), plot_edge_prob_results, yerr=plot_edge_prob_results_err,
                 color='b', fmt='--o')

    plt.ylabel('Edge Probability', fontsize=22)
    plt.xlabel('Triad Type', fontsize=22)
    plt.tight_layout()
    current_fig = plt.gcf()
    plt.xticks(np.arange(1, len(triangle_types) + 1), triangle_types)
    current_fig.savefig('{0}triad_edge_probability.pdf'.format(plot_save_path), format='pdf')
    plt.clf()


########## Link Prediction analysis ##############
def get_combined_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)).union(ego_net.predecessors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    return list(first_hop_nodes), list(second_hop_nodes), v_nodes


def get_specific_type_nodes(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node))

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)).union(ego_net.predecessors(z))) - first_hop_nodes

        # allowing types T01 and T02 and T05
        if ego_net.has_edge(z, ego_node):
            for v in temp_v_nodes:
                # this excludes #5
                # if v == ego_node or ego_net.has_edge(ego_node, v) or not ego_net.has_edge(z, v):
                if v == ego_node or ego_net.has_edge(ego_node, v):
                    continue

                if v not in v_nodes:
                    v_nodes[v] = [z]
                else:
                    v_nodes[v].append(z)
        # allowing type T03
        else:
            for v in temp_v_nodes:
                if v == ego_node or ego_net.has_edge(ego_node, v) or \
                        not (ego_net.has_edge(z, v) and ego_net.has_edge(v, z)):
                    continue

                if v not in v_nodes:
                    v_nodes[v] = [z]
                else:
                    v_nodes[v].append(z)

    return first_hop_nodes, v_nodes


def get_specific_type_nodes_gplus(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node))

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)).union(ego_net.predecessors(z))) - first_hop_nodes

        # allowing types T01 and T02
        if ego_net.has_edge(z, ego_node):
            for v in temp_v_nodes:
                if v == ego_node or ego_net.has_edge(ego_node, v) or not ego_net.has_edge(z, v):
                    continue

                if v not in v_nodes:
                    v_nodes[v] = [z]
                else:
                    v_nodes[v].append(z)
        # allowing type T06
        else:
            for v in temp_v_nodes:
                if v == ego_node or ego_net.has_edge(ego_node, v) or ego_net.has_edge(z, v):
                    continue

                if v not in v_nodes:
                    v_nodes[v] = [z]
                else:
                    v_nodes[v].append(z)

    # allowing types T07 and T09
    second_z_set = set(ego_net.predecessors(ego_node)) - first_hop_nodes
    for z in second_z_set:
        temp_v_nodes = set(ego_net.successors(z)) - first_hop_nodes

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v):
                continue

            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    return first_hop_nodes, v_nodes


def calc_top_k_scores(y_scores, y_true, top_k_values, percent_score):
    index_of_top_k_scores = np.argsort(y_scores)[::-1][:top_k_values[-1]]
    top_preds = y_true[index_of_top_k_scores]
    for k in top_k_values:
        if len(top_preds) >= k:
            percent_score[k].append(sum(top_preds[:k]) / k)


def run_directed_link_prediction(ego_net_file, top_k_values, data_file_base_path, result_file_base_path,
                                 specific_triads_only=False, skip_over_100k=True, skip_snapshots_w_no_new_edge=True):
    start_time = datetime.now()

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if skip_over_100k and os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    score_list = ['cn', 'dccn', 'aa', 'dcaa', 'car', 'cclp', 'dccar']
    percent_scores = {}

    for score in score_list:
        percent_scores[score] = {}
        for k in top_k_values:
            percent_scores[score][k] = []

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    total_y_true = 0

    num_nodes = nx.number_of_nodes(ego_net_snapshots[-1])
    # if the number of nodes in the last snapshot of the network is big, skip it and save a file in skipped-nets
    if skip_over_100k and num_nodes >= 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)
        return

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        if specific_triads_only:
            first_hop_nodes, v_nodes = get_specific_type_nodes_gplus(ego_net_snapshots[i], ego_node)
        else:
            first_hop_nodes, second_hop_nodes, v_nodes = get_combined_type_nodes(ego_net_snapshots[i], ego_node)

        v_nodes_list = list(v_nodes.keys())
        y_true = []

        for v_i in range(0, len(v_nodes_list)):
            if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                y_true.append(1)
            else:
                y_true.append(0)

        # if no edge was formed either skip it, or set all scores to 0, since that's what we are going to get.
        if np.sum(y_true) == 0:
            if skip_snapshots_w_no_new_edge:
                continue

            for s in score_list:
                for k in top_k_values:
                    if len(y_true) >= k:
                        percent_scores[s][k].append(0)
            continue

        total_y_true += np.sum(y_true)

        # numpy array is needed for sorting purposes
        y_true = np.array(y_true)

        # getting scores for cn, dccn, aa, dcaa
        lp_scores = aa_cn_dc_lp_scores_directed(ego_net_snapshots[i], v_nodes_list, v_nodes, first_hop_nodes)
        for s in lp_scores.keys():
            calc_top_k_scores(lp_scores[s], y_true, top_k_values, percent_scores[s])

        # getting scores for car and cclp
        lp_scores = car_dccar_cclp_directed_lp(ego_net_snapshots[i], v_nodes_list, v_nodes, first_hop_nodes)
        for s in lp_scores.keys():
            calc_top_k_scores(lp_scores[s], y_true, top_k_values, percent_scores[s])

    # skip if no snapshot returned a score
    if len(percent_scores[score_list[0]][top_k_values[0]]) > 0:
        # getting the mean of all snapshots for each score
        for s in percent_scores:
            for k in top_k_values:
                if len(percent_scores[s][k]) == 0:
                    percent_scores[s][k] = np.nan
                else:
                    percent_scores[s][k] = np.mean(percent_scores[s][k])

        with open(result_file_base_path + 'results/' + ego_net_file, 'wb') as f:
            pickle.dump(percent_scores, f, protocol=-1)

        print("Analyzed ego net: {0} - Duration: {1} - Num nodes: {2} - Formed: {3}"
              .format(ego_net_file, datetime.now() - start_time, num_nodes, total_y_true))

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)


def run_directed_link_prediction_w_separate_degree(ego_net_file, top_k_values, data_file_base_path,
                                                   result_file_base_path, specific_triads_only=False,
                                                   skip_over_100k=True, skip_snapshots_w_no_new_edge=True):
    start_time = datetime.now()

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if skip_over_100k and os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    score_list = ['od-dccn', 'in-dccn', 'od-aa', 'id-aa', 'od-dcaa', 'in-dcaa']
    percent_scores = {}

    for score in score_list:
        percent_scores[score] = {}
        for k in top_k_values:
            percent_scores[score][k] = []

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    total_y_true = 0

    num_nodes = nx.number_of_nodes(ego_net_snapshots[-1])
    # if the number of nodes in the last snapshot of the network is big, skip it and save a file in skipped-nets
    if skip_over_100k and num_nodes >= 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)
        return

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        if specific_triads_only:
            first_hop_nodes, v_nodes = get_specific_type_nodes_gplus(ego_net_snapshots[i], ego_node)
        else:
            first_hop_nodes, second_hop_nodes, v_nodes = get_combined_type_nodes(ego_net_snapshots[i], ego_node)

        v_nodes_list = list(v_nodes.keys())
        y_true = []

        for v_i in range(0, len(v_nodes_list)):
            if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                y_true.append(1)
            else:
                y_true.append(0)

        # if no edge was formed either skip it, or set all scores to 0, since that's what we are going to get.
        if np.sum(y_true) == 0:
            if skip_snapshots_w_no_new_edge:
                continue

            for s in score_list:
                for k in top_k_values:
                    if len(y_true) >= k:
                        percent_scores[s][k].append(0)
            continue

        total_y_true += np.sum(y_true)

        # numpy array is needed for sorting purposes
        y_true = np.array(y_true)

        # getting all scores
        lp_scores = aa_cn_dc_lp_w_separate_degree_directed(ego_net_snapshots[i], v_nodes_list, v_nodes, first_hop_nodes)
        for s in lp_scores.keys():
            calc_top_k_scores(lp_scores[s], y_true, top_k_values, percent_scores[s])

    # skip if no snapshot returned a score
    if len(percent_scores[score_list[0]][top_k_values[0]]) > 0:
        # getting the mean of all snapshots for each score
        for s in percent_scores:
            for k in top_k_values:
                if len(percent_scores[s][k]) == 0:
                    percent_scores[s][k] = np.nan
                else:
                    percent_scores[s][k] = np.mean(percent_scores[s][k])

        with open(result_file_base_path + 'results/' + ego_net_file, 'wb') as f:
            pickle.dump(percent_scores, f, protocol=-1)

        print("Analyzed ego net: {0} - Duration: {1} - Num nodes: {2} - Formed: {3}"
              .format(ego_net_file, datetime.now() - start_time, num_nodes, total_y_true))

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)


def aa_cn_dc_lp_w_separate_degree_directed(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):
    # this is the same function as `all_directed_lp_indices_combined` method, only returns a dict of scores instead.

    scores = {
        'od-dccn': [],
        'in-dccn': [],
        'od-aa': [],
        'id-aa': [],
        'od-dcaa': [],
        'in-dcaa': []
    }

    # a dict of info on z nodes. Every key points to a list [od-dccn, id-dccn, od-aa, in-aa, od-dcaa, id-dcaa]
    z_info = {}

    for z in first_hop_nodes:
        z_preds = set(ego_net.predecessors(z))
        z_succs = set(ego_net.successors(z))
        z_neighbors = z_preds.union(z_succs)

        # This should be the intersection of z_neighbors with the union of nodes in first and second hops
        z_global_degree = len(z_neighbors)
        z_global_in_degree = len(z_preds)
        z_global_out_degree = len(z_succs)

        z_local_degree = len(z_neighbors.intersection(first_hop_nodes))
        z_local_in_degree = len(z_preds.intersection(first_hop_nodes))
        z_local_out_degree = len(z_succs.intersection(first_hop_nodes))

        y = z_global_degree - z_local_degree

        # if y = 1, then the z node has no neighbor in the second hop, thus no need to compute
        if y == 1:
            continue

        z_info[z] = []
        # od-dccn
        z_info[z].append(math.log(z_local_out_degree + 2))

        # id-dccn
        z_info[z].append(math.log(z_local_in_degree + 2))

        # od-aa
        z_info[z].append(1 / math.log(z_global_out_degree + 2))

        # in-aa
        z_info[z].append(1 / math.log(z_global_in_degree + 2))

        # od-dcaa
        z_local_out_degree += 1
        z_global_out_degree += 2
        y_od = z_global_out_degree - z_local_out_degree
        od_dcaa = 1 / math.log((z_local_out_degree * (1 - (z_local_out_degree / z_global_out_degree))) +
                               (y_od * (z_global_out_degree / z_local_out_degree)))
        z_info[z].append(od_dcaa)

        # id-dcaa
        z_local_in_degree += 1
        z_global_in_degree += 2
        y_id = z_global_in_degree - z_local_in_degree
        id_dcaa = 1 / math.log((z_local_in_degree * (1 - (z_local_in_degree / z_global_in_degree))) +
                               (y_id * (z_global_in_degree / z_local_in_degree)))
        z_info[z].append(id_dcaa)

    for v_i in range(len(v_nodes_list)):
        temp_od_dccn = 0
        temp_id_dccn = 0
        temp_od_aa = 0
        temp_id_aa = 0
        temp_od_dcaa = 0
        temp_in_dcaa = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            if z not in z_info:
                z_info[z] = aa_cn_dc_lp_w_sep_degree_directed_for_single_node(ego_net, z, first_hop_nodes)

            temp_od_dccn += z_info[z][0]
            temp_id_dccn += z_info[z][1]
            temp_od_aa += z_info[z][2]
            temp_id_aa += z_info[z][3]
            temp_od_dcaa += z_info[z][4]
            temp_in_dcaa += z_info[z][5]

        scores['od-dccn'].append(temp_od_dccn)
        scores['in-dccn'].append(temp_id_dccn)
        scores['od-aa'].append(temp_od_aa)
        scores['id-aa'].append(temp_id_aa)
        scores['od-dcaa'].append(temp_od_dcaa)
        scores['in-dcaa'].append(temp_in_dcaa)

    return scores


def aa_cn_dc_lp_w_sep_degree_directed_for_single_node(ego_net, z, first_hop_nodes):
    z_info = []
    z_preds = set(ego_net.predecessors(z))
    z_succs = set(ego_net.successors(z))

    z_global_in_degree = len(z_preds)
    z_global_out_degree = len(z_succs)

    z_local_in_degree = len(z_preds.intersection(first_hop_nodes))
    z_local_out_degree = len(z_succs.intersection(first_hop_nodes))

    # od-dccn
    z_info.append(math.log(z_local_out_degree + 2))

    # id-dccn
    z_info.append(math.log(z_local_in_degree + 2))

    # od-aa
    z_info.append(1 / math.log(z_global_out_degree + 2))

    # in-aa
    z_info[z].append(1 / math.log(z_global_in_degree + 2))

    # od-dcaa
    z_local_out_degree += 1
    z_global_out_degree += 2
    y_od = z_global_out_degree - z_local_out_degree
    od_dcaa = 1 / math.log((z_local_out_degree * (1 - (z_local_out_degree / z_global_out_degree))) +
                           (y_od * (z_global_out_degree / z_local_out_degree)))
    z_info.append(od_dcaa)

    # id-dcaa
    z_local_in_degree += 1
    z_global_in_degree += 2
    y_id = z_global_in_degree - z_local_in_degree
    id_dcaa = 1 / math.log((z_local_in_degree * (1 - (z_local_in_degree / z_global_in_degree))) +
                           (y_id * (z_global_in_degree / z_local_in_degree)))
    z_info.append(id_dcaa)

    return z_info


def aa_cn_dc_lp_scores_directed(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):
    # this is the same function as `all_directed_lp_indices_combined` method, only returns a dict of scores instead.

    scores = {
        'cn': [],
        'dccn': [],
        'aa': [],
        'dcaa': []
    }

    # a dict of info on z nodes. Every key points to a list [dccn, aa, dcaa]
    z_info = {}

    for z in first_hop_nodes:
        z_neighbors = set(ego_net.predecessors(z)).union(set(ego_net.successors(z)))

        # This should be the intersection of z_neighbors with the union of nodes in first and second hops
        z_global_degree = len(z_neighbors)

        z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

        y = z_global_degree - z_local_degree

        # if y = 1, then the z node has no neighbor in the second hop, thus no need to compute
        if y == 1:
            continue

        z_info[z] = []
        # dccn
        # temp_dccn = (z_local_degree + len(v_nodes_z[v_nodes_list[v_i]]))
        z_info[z].append(math.log(z_local_degree + 2))

        # aa
        z_info[z].append(1 / math.log(z_global_degree))

        # dcaa
        z_local_degree += 1
        dcaa = 1 / math.log((z_local_degree * (1 - (z_local_degree / z_global_degree))) +
                            (y * (z_global_degree / z_local_degree)))

        z_info[z].append(dcaa)

    for v_i in range(len(v_nodes_list)):
        # cn score
        scores['cn'].append(len(v_nodes_z[v_nodes_list[v_i]]))

        temp_dccn = 0
        temp_aa = 0
        temp_dcaa = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            if z not in z_info:
                z_info[z] = aa_dc_lp_scores_directed_for_single_node(ego_net, z, first_hop_nodes)

            temp_dccn += z_info[z][0]
            temp_aa += z_info[z][1]
            temp_dcaa += z_info[z][2]

        scores['dccn'].append(temp_dccn)
        scores['aa'].append(temp_aa)
        scores['dcaa'].append(temp_dcaa)

    return scores


def aa_dc_lp_scores_directed_for_single_node(ego_net, z, first_hop_nodes):
    single_z_info = []

    z_neighbors = set(ego_net.predecessors(z)).union(set(ego_net.successors(z)))

    # This should be the intersection of z_neighbors with the union of nodes in first and second hops
    z_global_degree = len(z_neighbors)

    z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

    y = z_global_degree - z_local_degree

    # # if y = 1, then the z node has no neighbor in the second hop, thus no need to compute
    # if y == 1:
    #     continue

    # dccn
    # temp_dccn = (z_local_degree + len(v_nodes_z[v_nodes_list[v_i]]))
    single_z_info.append(math.log(z_local_degree + 2))

    # aa
    single_z_info.append(1 / math.log(z_global_degree))

    # dcaa
    z_local_degree += 1
    dcaa = 1 / math.log((z_local_degree * (1 - (z_local_degree / z_global_degree))) +
                        (y * (z_global_degree / z_local_degree)))

    single_z_info.append(dcaa)

    return single_z_info


def car_dccar_cclp_directed_lp(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):
    # exact same as `car_and_cclp_directed_lp_indices_combined`, but this returns a dict instead

    scores = {
        'car': [],
        'dccar': [],
        'cclp': []
    }

    undirected_ego_net = ego_net.to_undirected()

    z_info = {}

    for z in first_hop_nodes:
        z_deg = undirected_ego_net.degree(z)

        # if z_deg = 1, then the z node has no neighbor in the second hop, thus no need to compute
        if z_deg == 1:
            continue

        z_info[z] = []

        # calculation for LD-CAR
        z_local_degree = len(set(nx.neighbors(undirected_ego_net, z)).intersection(first_hop_nodes))
        z_info[z].append(math.log(z_local_degree + 2))

        # CCLP
        z_tri = nx.triangles(undirected_ego_net, z)
        z_info[z].append(z_tri / (z_deg * (z_deg - 1) / 2))

    for v_i in range(0, len(v_nodes_list)):
        num_cn = len(v_nodes_z[v_nodes_list[v_i]])
        lcl = 0

        if num_cn > 1:
            for zz_i in range(num_cn - 1):
                for zzz_i in range(zz_i + 1, num_cn):
                    if ego_net.has_edge(v_nodes_z[v_nodes_list[v_i]][zz_i], v_nodes_z[v_nodes_list[v_i]][zzz_i]) or \
                            ego_net.has_edge(v_nodes_z[v_nodes_list[v_i]][zzz_i], v_nodes_z[v_nodes_list[v_i]][zz_i]):
                        lcl += 1

        # car score
        scores['car'].append(num_cn * lcl)

        temp_cclp = 0
        temp_dccar = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            if z not in z_info:
                z_single_cclp, z_single_car_temp = cclp_dccn_directed_lp_single_node(undirected_ego_net, z,
                                                                                     first_hop_nodes)
                z_info[z].append(z_single_car_temp)
                z_info[z].append(z_single_cclp)

            temp_dccar += z_info[z][0]
            temp_cclp += z_info[z][1]

        # cclp and dccar score
        scores['cclp'].append(temp_cclp)
        scores['dccar'].append(temp_dccar * lcl)
    return scores


def cclp_dccn_directed_lp_single_node(undirected_ego_net, z, first_hop_nodes):
    z_deg = undirected_ego_net.degree(z)

    z_tri = nx.triangles(undirected_ego_net, z)

    z_single_cclp = z_tri / (z_deg * (z_deg - 1) / 2)

    z_local_degree = len(set(nx.neighbors(undirected_ego_net, z)).intersection(first_hop_nodes))
    z_single_car_temp = math.log(z_local_degree + 2)

    return z_single_cclp, z_single_car_temp


def dc_car_cclp_directed_lp(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):

    scores = {
        'dccar': [],
        'dccclp': []
    }

    undirected_ego_net = ego_net.to_undirected()

    # a dict of info on z nodes. Every key points to a list [dccar, dccclp]
    z_info = {}

    for z in first_hop_nodes:
        z_neighbors = set(ego_net.predecessors(z)).union(set(ego_net.successors(z)))

        # This should be the intersection of z_neighbors with the union of nodes in first and second hops
        z_global_degree = len(z_neighbors)

        z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

        y = z_global_degree - z_local_degree

        # if y = 1, then the z node has no neighbor in the second hop, thus no need to compute
        if y == 1:
            continue

        z_info[z] = []
        # dccar
        z_info[z].append(math.log(z_local_degree + 2))

        # aa
        z_info[z].append(1 / math.log(z_global_degree))

        # dcaa
        z_local_degree += 1
        dcaa = 1 / math.log((z_local_degree * (1 - (z_local_degree / z_global_degree))) +
                            (y * (z_global_degree / z_local_degree)))

        z_info[z].append(dcaa)

    for v_i in range(len(v_nodes_list)):
        lcl = undirected_ego_net.subgraph(v_nodes_z[v_nodes_list[v_i]]).number_of_edges()

        temp_dccar = 0
        temp_dccclp = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            temp_dccar += z_info[z][0]
            temp_dccclp += z_info[z][1]

        scores['dccar'].append(temp_dccar * lcl)
        scores['dccclp'].append(temp_dccclp)

    return scores


########## Link Prediction on Test Method ##############
def run_directed_link_prediciton_on_test_method(method_pointer, method_name, ego_net_file, top_k_values,
                                                data_file_base_path, result_file_base_path, skip_over_100k=True):
    start_time = datetime.now()

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if skip_over_100k and os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    percent_scores = {}
    percent_scores[method_name] = {}
    for k in top_k_values:
        percent_scores[method_name][k] = []

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    total_y_true = 0

    num_nodes = nx.number_of_nodes(ego_net_snapshots[-1])
    # if the number of nodes in the last snapshot of the network is big, skip it and save a file in skipped-nets
    if skip_over_100k and num_nodes >= 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)
        return

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes, second_hop_nodes, v_nodes = get_combined_type_nodes(ego_net_snapshots[i], ego_node)

        v_nodes_list = list(v_nodes.keys())
        y_true = []

        for v_i in range(0, len(v_nodes_list)):
            if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                y_true.append(1)
            else:
                y_true.append(0)

        # continue of no edge was formed
        if np.sum(y_true) == 0:
            continue

        total_y_true += np.sum(y_true)

        # numpy array is needed for sorting purposes
        y_true = np.array(y_true)

        # getting scores
        lp_scores = method_pointer(ego_net_snapshots[i], v_nodes_list, v_nodes, first_hop_nodes)
        calc_top_k_scores(lp_scores, y_true, top_k_values, percent_scores[method_name])

    # skip if no snapshot returned a score
    if len(percent_scores[method_name][top_k_values[0]]) > 0:
        # getting the mean of all snapshots for each score
        for k in top_k_values:
            percent_scores[method_name][k] = np.mean(percent_scores[method_name][k])

        with open(result_file_base_path + 'results/' + ego_net_file, 'wb') as f:
            pickle.dump(percent_scores, f, protocol=-1)

        print("Analyzed ego net: {0} - Duration: {1} - Num nodes: {2} - Formed: {3}"
              .format(ego_net_file, datetime.now() - start_time, num_nodes, total_y_true))

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)


def test1_lp_scores_directed(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):
    # Method description:

    scores = []
    # a dict of info on z nodes. Every key points to the test score
    z_info = {}

    for z in first_hop_nodes:
        z_neighbors = set(ego_net.predecessors(z)).union(set(ego_net.successors(z)))

        # This should be the intersection of z_neighbors with the union of nodes in first and second hops
        z_global_degree = len(z_neighbors)

        z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

        y = z_global_degree - z_local_degree

        # if y = 1, then the z node has no neighbor in the second hop, thus no need to compute
        if y == 1:
            continue

        z_local_degree += 1
        test_method = 1 / math.log((z_local_degree * (1 - (z_local_degree / z_global_degree))) +
                            (y * (z_global_degree / z_local_degree)))

        z_info[z] = test_method

    for v_i in range(len(v_nodes_list)):
        temp = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            temp += z_info[z]

        scores.append(temp)

    return scores


def dccar_test1(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):
    scores = []

    # undirected_ego_net = ego_net.to_undirected()

    # a dict of info on z nodes. Every key points to the z nodes dccn value
    z_info = {}

    for z in first_hop_nodes:
        z_neighbors = set(ego_net.predecessors(z)).union(set(ego_net.successors(z)))

        z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

        # dccn for dccar
        z_info[z] = math.log(z_local_degree + 2)

    for v_i in range(len(v_nodes_list)):
        lcl = 0

        if len(v_nodes_z[v_nodes_list[v_i]]) > 1:
            for zz_i in range(len(v_nodes_z[v_nodes_list[v_i]]) - 1):
                for zzz_i in range(zz_i + 1, len(v_nodes_z[v_nodes_list[v_i]])):
                    if ego_net.has_edge(v_nodes_z[v_nodes_list[v_i]][zz_i], v_nodes_z[v_nodes_list[v_i]][zzz_i]) or \
                            ego_net.has_edge(v_nodes_z[v_nodes_list[v_i]][zzz_i], v_nodes_z[v_nodes_list[v_i]][zz_i]):
                        lcl += 1

        # lcl2 = undirected_ego_net.subgraph(v_nodes_z[v_nodes_list[v_i]]).number_of_edges()

        temp_dccar = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            temp_dccar += z_info[z]

        scores.append(temp_dccar * lcl)

    return scores


def dccclp_test1(ego_net, v_nodes_list, v_nodes_z, first_hop_nodes):
    scores = []

    undirected_ego_net = ego_net.to_undirected()

    z_cclp = {}

    for z in first_hop_nodes:
        z_deg = undirected_ego_net.degree(z)

        # if z_deg = 1, then the z node has no neighbor in the second hop, thus no need to compute
        if z_deg == 1:
            continue

        z_neighbors = set(ego_net.predecessors(z)).union(set(ego_net.successors(z)))

        # This should be the intersection of z_neighbors with the union of nodes in first and second hops
        z_global_degree = len(z_neighbors)

        z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

        y = z_global_degree - z_local_degree

        # if y = 1, then the z node has no neighbor in the second hop, thus no need to compute
        if y == 1:
            continue

        # dcaa
        z_tri = nx.triangles(undirected_ego_net, z)
        z_cclp[z] = (z_tri + z_local_degree + 1) / (((z_deg + len(first_hop_nodes)) * (z_deg + len(first_hop_nodes) - 1) / 2))

        # z_cclp[z] = z_tri / ((z_local_degree * (1 - (z_local_degree / z_global_degree))) +
        #                      (y * (z_global_degree / z_local_degree)))


        # z_tri = nx.triangles(undirected_ego_net, z)
        # z_y = z_deg - z_local_degree
        # z_cclp[z] = z_tri / (z_y * (z_y - 1) / 2)

    for v_i in range(0, len(v_nodes_list)):
        temp_cclp = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            temp_cclp += z_cclp[z]

        scores.append(temp_cclp)

    return scores


def get_combined_type_nodes_with_triad_ratio(ego_net, ego_node):
    first_hop_nodes = set(ego_net.successors(ego_node))
    second_hop_nodes = set()

    v_nodes = {}

    for z in first_hop_nodes:
        temp_v_nodes = (set(ego_net.successors(z)).union(ego_net.predecessors(z))) - first_hop_nodes
        second_hop_nodes = second_hop_nodes.union(temp_v_nodes)

        for v in temp_v_nodes:
            if v == ego_node or ego_net.has_edge(ego_node, v):
                continue
            if v not in v_nodes:
                v_nodes[v] = [z]
            else:
                v_nodes[v].append(z)

    if ego_node in second_hop_nodes:
        second_hop_nodes.remove(ego_node)

    # triads are the same as all, but ending with 0 and 1 is with and without an edge from v to u.
    triad_radio = {'T010': 0, 'T011': 0, 'T020': 0, 'T021': 0, 'T030': 0, 'T031': 0, 'T040': 0, 'T041': 0, 'T050': 0,
                   'T051': 0, 'T060': 0, 'T061': 0}

    for z1 in first_hop_nodes:
        for z2 in first_hop_nodes:
            if z1 == z2:
                continue

            if ego_net.has_edge(z1, ego_node):
                if ego_net.has_edge(z1, z2):
                    if ego_net.has_edge(z2, z1):
                        if ego_net.has_edge(z2, ego_node):
                            triad_radio['T011'] += 1
                        else:
                            triad_radio['T010'] += 1
                    else:
                        if ego_net.has_edge(z2, ego_node):
                            triad_radio['T021'] += 1
                        else:
                            triad_radio['T020'] += 1
                else:
                    if ego_net.has_edge(z2, ego_node):
                        triad_radio['T051'] += 1
                    else:
                        triad_radio['T050'] += 1
            else:
                if ego_net.has_edge(z1, z2):
                    if ego_net.has_edge(z2, z1):
                        if ego_net.has_edge(z2, ego_node):
                            triad_radio['T031'] += 1
                        else:
                            triad_radio['T030'] += 1
                    else:
                        if ego_net.has_edge(z2, ego_node):
                            triad_radio['T041'] += 1
                        else:
                            triad_radio['T040'] += 1
                else:
                    if ego_net.has_edge(z2, ego_node):
                        triad_radio['T061'] += 1
                    else:
                        triad_radio['T060'] += 1

    total = 0
    for t_type in triad_radio.keys():
        total += triad_radio[t_type]

    if total != 0:
        for t_type in triad_radio.keys():
            if triad_radio[t_type] == 0:
                triad_radio[t_type] = 0.01
                continue

            triad_radio[t_type] = triad_radio[t_type] / total
    else:
        for t_type in triad_radio.keys():
            triad_radio[t_type] = 1

    return first_hop_nodes, second_hop_nodes, v_nodes, triad_radio


def cn_aa_car_cclp_on_triad_ratio(ego_net, ego_node, v_nodes_list, v_nodes_z, first_hop_nodes, triad_radio):
    # this is the same function as `all_directed_lp_indices_combined` method, only returns a dict of scores instead.

    scores = {
        'cn': [],
        'aa': [],
        'car': [],
        'cclp': []
    }

    undirected_ego_net = ego_net.to_undirected()

    # a dict of info on z nodes. Every key points to a list [aa, cclp]
    z_info = {}

    for z in first_hop_nodes:
        z_info[z] = []

        z_neighbors = set(ego_net.predecessors(z)).union(ego_net.successors(z))
        z_global_degree = len(z_neighbors)

        # if global degree is 1, then z node has no neighbor in the second hop
        if z_global_degree == 1:
            continue
        # aa
        z_info[z].append(1 / math.log(z_global_degree))

        z_deg = undirected_ego_net.degree(z)
        z_tri = nx.triangles(undirected_ego_net, z)

        #cclp
        z_info[z].append(z_tri / (z_deg * (z_deg - 1) / 2))

    for v_i in range(len(v_nodes_list)):
        lcl = undirected_ego_net.subgraph(v_nodes_z[v_nodes_list[v_i]]).number_of_edges()
        v = v_nodes_list[v_i]

        temp_cn = 0
        temp_aa = 0
        temp_cclp = 0

        for z in v_nodes_z[v_nodes_list[v_i]]:
            triad_ratio_multiplier = -1
            if ego_net.has_edge(z, ego_node):
                if ego_net.has_edge(z, v):
                    if ego_net.has_edge(v, z):
                        if ego_net.has_edge(v, ego_node):
                            triad_ratio_multiplier = triad_radio['T011']
                        else:
                            triad_ratio_multiplier = triad_radio['T010']
                    else:
                        if ego_net.has_edge(v, ego_node):
                            triad_ratio_multiplier = triad_radio['T021']
                        else:
                            triad_ratio_multiplier = triad_radio['T020']
                else:
                    if ego_net.has_edge(v, ego_node):
                        triad_ratio_multiplier = triad_radio['T051']
                    else:
                        triad_ratio_multiplier = triad_radio['T050']
            else:
                if ego_net.has_edge(z, v):
                    if ego_net.has_edge(v, z):
                        if ego_net.has_edge(v, ego_node):
                            triad_ratio_multiplier = triad_radio['T031']
                        else:
                            triad_ratio_multiplier = triad_radio['T030']
                    else:
                        if ego_net.has_edge(v, ego_node):
                            triad_ratio_multiplier = triad_radio['T041']
                        else:
                            triad_ratio_multiplier = triad_radio['T040']
                else:
                    if ego_net.has_edge(v, ego_node):
                        triad_ratio_multiplier = triad_radio['T061']
                    else:
                        triad_ratio_multiplier = triad_radio['T060']

            if triad_ratio_multiplier == -1:
                print("triad ratio is wrong")

            temp_cn += triad_ratio_multiplier
            temp_aa += z_info[z][0] * triad_ratio_multiplier
            temp_cclp += z_info[z][1] * triad_ratio_multiplier

        scores['cn'].append(temp_cn)
        scores['aa'].append(temp_aa)
        scores['car'].append(temp_cn * lcl)
        scores['cclp'].append(temp_cclp)

    return scores


def run_directed_link_prediction_on_personalized_tirad(ego_net_file, top_k_values, data_file_base_path,
                                                       result_file_base_path, skip_over_100k=True):
    start_time = datetime.now()

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if skip_over_100k and os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    score_list = ['cn', 'aa', 'car', 'cclp']
    percent_scores = {}

    for score in score_list:
        percent_scores[score] = {}
        for k in top_k_values:
            percent_scores[score][k] = []

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    total_y_true = 0

    num_nodes = nx.number_of_nodes(ego_net_snapshots[-1])
    # if the number of nodes in the last snapshot of the network is big, skip it and save a file in skipped-nets
    if skip_over_100k and num_nodes >= 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)
        return

    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes, second_hop_nodes, v_nodes, triad_ratio = \
            get_combined_type_nodes_with_triad_ratio(ego_net_snapshots[i], ego_node)

        v_nodes_list = list(v_nodes.keys())
        y_true = []

        for v_i in range(0, len(v_nodes_list)):
            if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]):
                y_true.append(1)
            else:
                y_true.append(0)

        # continue of no edge was formed
        if np.sum(y_true) == 0:
            continue

        total_y_true += np.sum(y_true)

        # numpy array is needed for sorting purposes
        y_true = np.array(y_true)

        # getting scores for cn, aa, car, cclp
        lp_scores = cn_aa_car_cclp_on_triad_ratio(ego_net_snapshots[i], ego_node, v_nodes_list, v_nodes,
                                                  first_hop_nodes, triad_ratio)
        for s in lp_scores.keys():
            calc_top_k_scores(lp_scores[s], y_true, top_k_values, percent_scores[s])

    # skip if no snapshot returned a score
    if len(percent_scores[score_list[0]][top_k_values[0]]) > 0:
        # getting the mean of all snapshots for each score
        for s in percent_scores:
            for k in top_k_values:
                percent_scores[s][k] = np.mean(percent_scores[s][k])

        with open(result_file_base_path + 'results/' + ego_net_file, 'wb') as f:
            pickle.dump(percent_scores, f, protocol=-1)

        print("Analyzed ego net: {0} - Duration: {1} - Num nodes: {2} - Formed: {3}"
              .format(ego_net_file, datetime.now() - start_time, num_nodes, total_y_true))

    # save an empty file in analyzed_egonets to know which ones were analyzed
    with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
        pickle.dump(0, f, protocol=-1)

