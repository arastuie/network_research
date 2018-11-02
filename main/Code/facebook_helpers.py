import os
import math
import time
import pickle
import warnings
import numpy as np
import helpers as h
import networkx as nx
import matplotlib.pyplot as plt
import link_prediction_evaluator as lpe


dataset_file_path = '/shared/DataSets/FacebookViswanath2009/raw/facebook-links.txt'
egonet_files_path = '/shared/DataSets/FacebookViswanath2009/egocentric/all_egonets/'

empirical_pickle_base_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/'
empirical_pickle_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/pickle-files-1/'
empirical_plot_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/plots-1/'

# lp_results_path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files-1/'
# lp_plots_path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/plots-1/'

lp_results_path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/pickle-files-2/'
lp_plots_path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb/plots-2/'

pymk_directories = ['before-pymk', 'after-pymk']


# ********** Reading facebook data ********** #
def read_graph():
    print("Reading the original graph...")

    t0 = time.time()
    original_graph = nx.Graph()

    with open(dataset_file_path, 'r') as infile:
        for l in infile:
            nums = l.rstrip().split("\t")

            # replace no timestamp with -1
            if nums[2] == '\\N':
                nums[2] = -1

            original_graph.add_edge(int(nums[0]), int(nums[1]), timestamp=int(nums[2]))

    print("Facebook network in. Took {0:.2f}min".format((time.time() - t0) / 60))
    return original_graph


def extract_all_ego_centric_networks_in_fb(original_graph):
    """
    Extracts and saves all ego centric networks, divided into 10 snapshots

    :param original_graph: The original graph containing all nodes and edges
    """
    print()

    orig_snapshots = []
    first_timestamp = 1157454929
    seconds_in_90_days = 90 * 24 * 60 * 60

    for i in range(1, 11):
        orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in original_graph.edges(data=True)
                                        if d['timestamp'] < (first_timestamp + seconds_in_90_days * i)]))

    extracted_nodes = set()

    # Nodes appeared only in the last snapshots do not need to be extracted
    for i in range(len(orig_snapshots) - 1):
        nodes_to_extract = set(orig_snapshots[i].nodes()) - extracted_nodes

        for ego in nodes_to_extract:
            ego_snapshots = []

            for u in range(i, len(orig_snapshots)):
                ego_snapshots.append(nx.ego_graph(orig_snapshots[u], ego, radius=2))

            with open("{}{}.pckl".format(egonet_files_path, ego), 'wb') as f:
                pickle.dump([ego, ego_snapshots], f, protocol=-1)

            extracted_nodes.add(ego)
            print("Num nodes extracted: {0}".format(len(extracted_nodes)), end="\r")

    return


# ********** Local degree empirical analysis ********** #
def run_local_degree_empirical_analysis(ego_net_file, results_base_path, log_degree, skip_snaps, normalize):
    # This analysis is separated into befor and after PYMK. FB intorduced PYMK around March 2008, which means
    # links created between snapshots 0 to 5 are for before PYMK and the ones made between 5 to 9 are after.
    # Also since not all ego nets have 10 snapshots, the index for before and after must be first calculated.

    # log_degee (boolean): take log of the degrees before the mean
    # skip_snaps (boolean): skip snapshots where there is no edges formed
    # normalize (boolean): normalize the mean of each snapshots

    # return if the egonet is on the analyzed list in after pymk, it must be in before as well
    if os.path.isfile(results_base_path + 'after-pymk/analyzed_egonets/' + ego_net_file):
        return

    with open(egonet_files_path + ego_net_file, 'rb') as f:
        ego, ego_snaps = pickle.load(f)

    num_snaps = len(ego_snaps)

    for pymk_type in pymk_directories:
        if pymk_type == 'before-pymk':
            # at least need 6 snapshots to have 1 snapshot jump for before PYMK
            before_pymk_ending_snap = num_snaps - 5
            if before_pymk_ending_snap < 1:
                continue

            start = 0
            end = before_pymk_ending_snap

        else:
            after_pymk_starting_snap = num_snaps - 5
            if after_pymk_starting_snap < 0:
                after_pymk_starting_snap = 0

            start = after_pymk_starting_snap
            end = num_snaps - 1

        snapshots_local_formed = []
        snapshots_global_formed = []
        snapshots_local_not_formed = []
        snapshots_global_not_formed = []

        for i in range(start, end):
            # Get a list of v nodes which did or did not form an edge with the ego in the next snapshot
            current_snap_z_nodes = set(ego_snaps[i].neighbors(ego))
            current_snap_v_nodes = set(ego_snaps[i].nodes()) - current_snap_z_nodes
            current_snap_v_nodes.remove(ego)

            next_snap_z_nodes = set(ego_snaps[i + 1].neighbors(ego))

            formed_v_nodes = next_snap_z_nodes.intersection(current_snap_v_nodes)
            not_formed_v_nodes = current_snap_v_nodes - formed_v_nodes

            if skip_snaps and (len(formed_v_nodes) == 0 or len(not_formed_v_nodes) == 0):
                continue

            len_first_hop = len(current_snap_z_nodes)
            tot_num_nodes = ego_snaps[i].number_of_nodes()

            z_local_formed = []
            z_global_formed = []
            z_local_not_formed = []
            z_global_not_formed = []

            # Getting degrees of the z nodes to speed up the process
            z_local_degree = {}
            z_global_degree = {}
            for z in current_snap_z_nodes:
                if log_degree:
                    z_local_degree[z] = np.log(len(list(nx.common_neighbors(ego_snaps[i], z, ego))) + 2)
                    z_global_degree[z] = np.log(nx.degree(ego_snaps[i], z) + 2)
                else:
                    z_local_degree[z] = len(list(nx.common_neighbors(ego_snaps[i], z, ego)))
                    z_global_degree[z] = nx.degree(ego_snaps[i], z)

            for v in formed_v_nodes:
                common_neighbors = nx.common_neighbors(ego_snaps[i], v, ego)
                temp_local = []
                temp_global = []

                for cn in common_neighbors:
                    temp_local.append(z_local_degree[cn])
                    temp_global.append(z_global_degree[cn])

                z_local_formed.append(np.mean(temp_local))
                z_global_formed.append(np.mean(temp_global))

            for v in not_formed_v_nodes:
                common_neighbors = list(nx.common_neighbors(ego_snaps[i], v, ego))
                temp_local = []
                temp_global = []

                for cn in common_neighbors:
                    temp_local.append(z_local_degree[cn])
                    temp_global.append(z_global_degree[cn])

                z_local_not_formed.append(np.mean(temp_local))
                z_global_not_formed.append(np.mean(temp_global))

            # if any of these arrays are empty due to no formed edges, the mean returns a nan value, which is what
            # we want. So ignore the warning.
            if normalize:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    snapshots_local_formed.append(np.mean(z_local_formed) / len_first_hop)
                    snapshots_global_formed.append(np.mean(z_global_formed) / tot_num_nodes)
                    snapshots_local_not_formed.append(np.mean(z_local_not_formed) / len_first_hop)
                    snapshots_global_not_formed.append(np.mean(z_global_not_formed) / tot_num_nodes)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    snapshots_local_formed.append(np.mean(z_local_formed))
                    snapshots_global_formed.append(np.mean(z_global_formed))
                    snapshots_local_not_formed.append(np.mean(z_local_not_formed))
                    snapshots_global_not_formed.append(np.mean(z_global_not_formed))

        if len(snapshots_local_formed) > 0:
            # if any of these arrays are all nans due to no formed edges in all snapshots, the nanmean returns a nan
            # value, which is what we want. So ignore the warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                with open(results_base_path + pymk_type + '/results/' + ego_net_file, 'wb') as f:
                    pickle.dump([np.mean(snapshots_local_formed),
                                 np.mean(snapshots_global_formed),
                                 np.mean(snapshots_local_not_formed),
                                 np.mean(snapshots_global_not_formed)], f, protocol=-1)

        # save an empty file in analyzed_egonets to know which ones were analyzed
        with open(results_base_path + pymk_type + '/analyzed-egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)

    print("Analyzed ego net {0}".format(ego_net_file))
    return


def plot_local_degree_empirical_results(result_file_base_path, plot_save_path, gather_individual_results=False):
    # Plotting info
    gl_labels = ['local', 'global']
    z = 1.96
    bar_width = 0.40
    opacity = 0.6
    error_config = {'ecolor': '0.3', 'capsize': 4, 'lw': 3, 'capthick': 2}
    bar_legends = ['Formed', 'Not Formed']
    dif_results_for_plotting = ['formed', 'not-formed']
    names = ['Before PYMK', 'After PYMK']
    bar_color = ['r', 'b']

    # The order must be kept, since the saved pickle files are in the same order
    result_types = ['local-formed', 'global-formed', 'local-not-formed', 'global-not-formed']

    all_results = {'before-pymk': {}, 'after-pymk': {}}

    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    for pymk_type in pymk_directories:
        if gather_individual_results:

            for res_type in result_types:
                all_results[pymk_type][res_type] = 0
                all_results[pymk_type][res_type + '-err'] = 0

            results = []

            for result_file in os.listdir(result_file_base_path + pymk_type + '/results'):
                with open(result_file_base_path + pymk_type + '/results/' + result_file, 'rb') as f:
                    egonet_result = pickle.load(f)
                    results.append(egonet_result)
            results = np.array(results)

            for i, res_type in enumerate(result_types):
                all_results[pymk_type][res_type] = np.nanmean(results[:, i])
                all_results[pymk_type][res_type + '-err'] = h.get_mean_ci(results[:, i], z, has_nan=True)

            # Create directory if not exists
            if not os.path.exists(result_file_base_path + pymk_type + '/all-scores'):
                os.makedirs(result_file_base_path + pymk_type + '/all-scores')

            with open(result_file_base_path + pymk_type + '/all-scores/all-barplot.pckl', 'wb') as f:
                pickle.dump(all_results[pymk_type], f, protocol=-1)

        else:
            with open(result_file_base_path + pymk_type + '/all-scores/all-barplot.pckl', 'rb') as f:
                all_results[pymk_type] = pickle.load(f)

    # plotting
    plt.rcParams["figure.figsize"] = (8, 10)
    for i_degree in gl_labels:
        plt.rc('legend', fontsize=25)
        plt.rc('xtick', labelsize=28)
        plt.rc('ytick', labelsize=14)

        fig, ax = plt.subplots()

        for i_bar in range(len(dif_results_for_plotting)):
            plt.bar(np.arange(len(names)) + bar_width * i_bar,
                    [all_results['before-pymk'][i_degree + '-' + dif_results_for_plotting[i_bar]],
                     all_results['after-pymk'][i_degree + '-' + dif_results_for_plotting[i_bar]]],
                    bar_width,
                    alpha=opacity,
                    color=bar_color[i_bar],
                    yerr=[all_results['before-pymk'][i_degree + '-' + dif_results_for_plotting[i_bar] + '-err'],
                          all_results['after-pymk'][i_degree + '-' + dif_results_for_plotting[i_bar] + '-err']],
                    error_kw=error_config,
                    label=bar_legends[i_bar])

        plt.ylabel('Mean Normalized {0} Degree'.format(i_degree.capitalize()), fontsize=25)
        plt.xticks(np.arange(len(names)) + bar_width / 2, names)
        plt.legend(loc='upper left')
        plt.tight_layout()
        # plt.ylim(ymax=0.4)

        plt.savefig('{0}barplot-{1}.pdf'.format(plot_save_path, i_degree), format='pdf')
        plt.clf()


def plot_local_degree_empirical_ecdf(result_file_base_path, plot_save_path, gather_individual_results=False):
    plot_save_path += 'cdf/'

    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    # The order must be kept, since the saved pickle files are in the same order
    result_types = ['local-formed', 'global-formed', 'local-not-formed', 'global-not-formed']
    for pymk_type in pymk_directories:
        if gather_individual_results:
            all_results = {}

            for res_type in result_types:
                all_results[res_type] = []

            results = []

            for result_file in os.listdir(result_file_base_path + pymk_type + '/results'):
                with open(result_file_base_path + pymk_type + '/results/' + result_file, 'rb') as f:
                    egonet_result = pickle.load(f)
                    results.append(egonet_result)
            results = np.array(results)

            for i, res_type in enumerate(result_types):
                all_results[res_type] = results[:, i]

            # Create directory if not exists
            if not os.path.exists(result_file_base_path + pymk_type + '/all-scores'):
                os.makedirs(result_file_base_path + pymk_type + '/all-scores')

            with open(result_file_base_path + pymk_type + '/all-scores/all-ecdf.pckl', 'wb') as f:
                pickle.dump(all_results, f, protocol=-1)

        else:
            with open(result_file_base_path + pymk_type + '/all-scores/all-ecdf.pckl', 'rb') as f:
                all_results = pickle.load(f)

        for dt in ['Global', 'Local']:
            plt.rc('legend', fontsize=20)
            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)

            h.add_ecdf_with_band_plot_undirected(all_results[dt.lower() + '-formed'], 'Formed', 'r')

            h.add_ecdf_with_band_plot_undirected(all_results[dt.lower() + '-not-formed'], 'Not Formed', 'b')

            plt.ylabel('Empirical CDF', fontsize=20)
            plt.xlabel('Mean Normalized {0} Degree'.format(dt), fontsize=20)
            plt.legend(loc='lower right')
            plt.tight_layout()
            current_fig = plt.gcf()
            current_fig.savefig('{0}ecdf-{1}-{2}.pdf'.format(plot_save_path, pymk_type, dt), format='pdf')
            plt.clf()


# ********** Local degree distribution analysis ********** #
def gather_local_degree_data(ego_net_file, results_base_path):
    # This analysis is setup the same way as empirical analysis

    if os.path.isfile(results_base_path + 'after-pymk/results/' + ego_net_file):
        return

    with open(egonet_files_path + ego_net_file, 'rb') as f:
        ego, ego_snaps = pickle.load(f)

    num_snaps = len(ego_snaps)

    for pymk_type in pymk_directories:
        if pymk_type == 'before-pymk':
            # at least need 5 snapshots to have 1 snapshot for before PYMK (this differs from other analyses)
            before_pymk_ending_snap = num_snaps - 4
            if before_pymk_ending_snap < 1:
                continue

            start = 0
            end = before_pymk_ending_snap

        else:
            after_pymk_starting_snap = num_snaps - 4
            if after_pymk_starting_snap < 0:
                after_pymk_starting_snap = 0

            start = after_pymk_starting_snap
            end = num_snaps

        # z_local_degree will contain a list from each snapshot of z local degrees
        z_local_degrees = []

        for i in range(start, end):
            snap_ctr = len(z_local_degrees)
            z_local_degrees.append([])

            for z in ego_snaps[i].neighbors(ego):
                z_local_degrees[snap_ctr].append(len(list(nx.common_neighbors(ego_snaps[i], z, ego))))

        with open(results_base_path + pymk_type + '/results/' + ego_net_file, 'wb') as f:
            pickle.dump([ego, z_local_degrees], f, protocol=-1)

    print("Analyzed ego net {0}".format(ego_net_file))
    return


# ********** Link prediction analysis ********** #
def calc_top_k_scores(y_scores, y_true, top_k_values, percent_score):
    index_of_top_k_scores = np.argsort(y_scores)[::-1][:top_k_values[-1]]
    top_preds = y_true[index_of_top_k_scores]
    for k in top_k_values:
        if len(top_preds) >= k:
            percent_score[k].append(sum(top_preds[:k]) / k)


def common_neighbors_index(ego_net, non_edges):
    scores = []

    for u, v in non_edges:
        scores.append(len(list(nx.common_neighbors(ego_net, u, v))))

    return scores


def degree_corrected_common_neighbors_index(ego_net, non_edges, first_hop_nodes):
    scores = []

    z_dccn = {}
    for z in first_hop_nodes:
        cn_neighbors = set(nx.neighbors(ego_net, z))

        # local degree
        l = len(cn_neighbors.intersection(first_hop_nodes)) + 2
        z_dccn[z] = math.log(l)

    for u, v in non_edges:
        dccn_score = 0
        common_neighbors = nx.common_neighbors(ego_net, u, v)

        for z in common_neighbors:
            dccn_score += z_dccn[z]

        scores.append(dccn_score)

    return scores


def degree_corrected_adamic_adar_index(ego_net, non_edges, first_hop_nodes):
    scores = []

    z_dcaa = {}
    for z in first_hop_nodes:
        cn_neighbors = set(nx.neighbors(ego_net, z))
        # total degree
        g = len(cn_neighbors)
        # local degree
        l = len(cn_neighbors.intersection(first_hop_nodes))

        y = g - l

        # if y == 1, z does not have an edge with v nodes
        if y == 1:
            continue

        l += 1
        z_dcaa[z] = 1 / math.log((l * (1 - l / g)) + (y * (g / l)))

    for u, v in non_edges:
        dcaa_score = 0
        common_neighbors = nx.common_neighbors(ego_net, u, v)

        for z in common_neighbors:
            dcaa_score += z_dcaa[z]

        scores.append(dcaa_score)

    return scores


def cclp(ego_net, non_edges, first_hop_nodes):
    scores = []

    z_cclp = {}
    for z in first_hop_nodes:
        z_tri = nx.triangles(ego_net, z)
        z_deg = ego_net.degree(z)

        # if z_deg is 1, it does not have edge with v nodes
        if z_deg == 1:
            continue

        z_cclp[z] = z_tri / (z_deg * (z_deg - 1) / 2)

    for u, v in non_edges:
        cclp_score = 0
        common_neighbors = nx.common_neighbors(ego_net, u, v)

        for z in common_neighbors:
            cclp_score += z_cclp[z]

        scores.append(cclp_score)

    return scores


def car(ego_net, non_edges):
    scores = []

    for u, v in non_edges:
        common_neighbors = list(nx.common_neighbors(ego_net, u, v))
        cc_sub_g = ego_net.subgraph(common_neighbors)

        scores.append(len(common_neighbors) * cc_sub_g.number_of_edges())

    return scores


def dccar(ego_net, non_edges, first_hop_nodes):
    scores = []

    z_dccn = {}
    for z in first_hop_nodes:
        cn_neighbors = set(nx.neighbors(ego_net, z))

        # local degree
        l = len(cn_neighbors.intersection(first_hop_nodes)) + 2
        z_dccn[z] = math.log(l)

    for u, v in non_edges:
        dccn_score = 0
        common_neighbors = list(nx.common_neighbors(ego_net, u, v))
        cc_sub_g = ego_net.subgraph(common_neighbors)

        for z in common_neighbors:
            dccn_score += z_dccn[z]

        scores.append(dccn_score * cc_sub_g.number_of_edges())

    return scores


def dccclp(ego_net, non_edges, first_hop_nodes):
    scores = []

    z_dccclp = {}
    for z in first_hop_nodes:
        z_tri = nx.triangles(ego_net, z)

        cn_neighbors = set(nx.neighbors(ego_net, z))
        # total degree
        z_deg = len(cn_neighbors)
        # local degree
        z_local_deg = len(cn_neighbors.intersection(first_hop_nodes))

        # if True, z does not have an edge with v nodes
        if z_deg - z_local_deg == 1:
            continue

        z_dccclp[z] = (z_tri + z_local_deg + 1) / \
                      ((z_deg + len(first_hop_nodes)) * ((z_deg + len(first_hop_nodes)) - 1) / 2)

    for u, v in non_edges:
        dccclp_score = 0
        common_neighbors = nx.common_neighbors(ego_net, u, v)

        for z in common_neighbors:
            dccclp_score += z_dccclp[z]

        scores.append(dccclp_score)

    return scores


def run_link_prediction_analysis(ego_net_file, top_k_values):
    start_time = time.time()

    # return if the egonet is on the analyzed list
    if os.path.isfile(lp_results_path + 'after-pymk/analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if os.path.isfile(lp_results_path + 'after-pymk/skipped_egonets/' + ego_net_file):
        return

    with open(egonet_files_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    num_snaps = len(ego_net_snapshots)

    for pymk_type in pymk_directories:
        if pymk_type == 'before-pymk':
            # at least need 6 snapshots to have 1 snapshot jump for before PYMK
            before_pymk_ending_snap = num_snaps - 5
            if before_pymk_ending_snap < 1:
                continue

            start = 0
            end = before_pymk_ending_snap

        else:
            after_pymk_starting_snap = num_snaps - 5
            if after_pymk_starting_snap < 0:
                after_pymk_starting_snap = 0

            start = after_pymk_starting_snap
            end = num_snaps - 1

        score_list = ['cn', 'dccn', 'aa', 'dcaa', 'car', 'dccar', 'cclp', 'dccclp']
        percent_scores = {}

        for score in score_list:
            percent_scores[score] = {}
            for k in top_k_values:
                percent_scores[score][k] = []

        total_y_true = 0

        num_nodes = nx.number_of_nodes(ego_net_snapshots[-1])

        for i in range(start, end):
            first_hop_nodes = set(ego_net_snapshots[i].neighbors(ego_node))

            if len(first_hop_nodes) == 0:
                continue

            second_hop_nodes = set(ego_net_snapshots[i].nodes()) - first_hop_nodes
            second_hop_nodes.remove(ego_node)

            formed_nodes = second_hop_nodes.intersection(ego_net_snapshots[i + 1].neighbors(ego_node))

            # if len(formed_nodes) == 0:
            #     continue

            non_edges = []
            y_true = []

            for n in second_hop_nodes:
                # adding node with no edge as tuple
                non_edges.append((ego_node, n))

                if n in formed_nodes:
                    y_true.append(1)
                else:
                    y_true.append(0)

            # numpy array is needed for sorting purposes
            y_true = np.array(y_true)

            # if no edge was formed, all scores will be zero
            num_edges_formed = np.sum(y_true)
            if num_edges_formed == 0:
                for s in score_list:
                    for k in top_k_values:
                        if len(y_true) >= k:
                            percent_scores[s][k].append(0)
                continue

            total_y_true += num_edges_formed

            # evaluating different link prediction methods
            if 'cn' in percent_scores:
                y_scores = common_neighbors_index(ego_net_snapshots[i], non_edges)
                calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['cn'])

            if 'dccn' in percent_scores:
                y_scores = degree_corrected_common_neighbors_index(ego_net_snapshots[i], non_edges, first_hop_nodes)
                calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['dccn'])

            if 'aa' in percent_scores:
                y_scores = [p for u, v, p in nx.adamic_adar_index(ego_net_snapshots[i], non_edges)]
                calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['aa'])

            if 'dcaa' in percent_scores:
                y_scores = degree_corrected_adamic_adar_index(ego_net_snapshots[i], non_edges, first_hop_nodes)
                calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['dcaa'])

            if 'car' in percent_scores:
                y_scores = car(ego_net_snapshots[i], non_edges)
                calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['car'])

            if 'dccar' in percent_scores:
                y_scores = dccar(ego_net_snapshots[i], non_edges, first_hop_nodes)
                calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['dccar'])

            if 'cclp' in percent_scores:
                y_scores = cclp(ego_net_snapshots[i], non_edges, first_hop_nodes)
                calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['cclp'])

            if 'dccclp' in percent_scores:
                y_scores = dccclp(ego_net_snapshots[i], non_edges, first_hop_nodes)
                calc_top_k_scores(y_scores, y_true, top_k_values, percent_scores['dccclp'])

        # skip if no snapshot returned a score
        if len(percent_scores[score_list[0]][top_k_values[0]]) > 0:
            # getting the mean of all snapshots for each score
            for s in percent_scores:
                for k in top_k_values:
                    if len(percent_scores[s][k]) == 0:
                        percent_scores[s][k] = np.nan
                    else:
                        percent_scores[s][k] = np.mean(percent_scores[s][k])

            with open(lp_results_path + pymk_type + '/results/' + ego_net_file, 'wb') as f:
                pickle.dump(percent_scores, f, protocol=-1)

            print("{} ego net: {} - Duration: {} - Num nodes: {} - Formed: {}"
                  .format(pymk_type, ego_net_file, time.time() - start_time, num_nodes, total_y_true))

        # save an empty file in analyzed_egonets to know which ones were analyzed
        with open(lp_results_path + pymk_type + '/analyzed_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)


def plot_percent_improvements(comparison_pairs, gather_individual_results=False):
    for pymk_type in pymk_directories:
        lpe.plot_percent_improvements(lp_results_path + pymk_type + '/', lp_plots_path, comparison_pairs,
                                      gather_individual_results, facebook_pymk_type=pymk_type)


def calculate_lp_performance(scores=None, gather_individual_results=False):
    for pymk_type in pymk_directories:
        print(pymk_type)
        lpe.calculate_lp_performance(lp_results_path + pymk_type + '/', scores=scores, is_fb=True,
                                     gather_individual_results=gather_individual_results)
