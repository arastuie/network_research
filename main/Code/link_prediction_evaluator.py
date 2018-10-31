import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

score_list = ['cn', 'dccn', 'aa', 'dcaa', 'car', 'cclp', 'dccar']
score_names = {'cn': 'CN', 'dccn': 'LD-CN', 'aa': 'AA', 'dcaa': 'LD-AA', 'car': 'CAR', 'cclp': 'CCLP',
               'dccar': 'LD-CAR'}

top_k_values = [1, 3, 5, 10, 15, 20, 25, 30]


def get_all_results(result_file_base_path, gather_individual_results, scores=None):
    if not gather_individual_results:
        with open(result_file_base_path + "cumulated_results/all-methods-results.pckle", 'rb') as f:
            all_scores = pickle.load(f)

        return all_scores

    if scores is None:
        scores = score_list

    all_scores = {}

    for score in scores:
        all_scores[score] = {}
        for k in top_k_values:
            all_scores[score][k] = []

    num_files = len(os.listdir(result_file_base_path + 'results'))

    cnt = 0
    # loading result data
    for result_file in os.listdir(result_file_base_path + 'results'):
        try:
            with open(result_file_base_path + 'results/' + result_file, 'rb') as f:
                egonet_lp_results = pickle.load(f)
        except EOFError:
            os.remove(result_file_base_path + 'results/' + result_file)

        for k in top_k_values:
            for score in scores:
                if np.isnan(egonet_lp_results[score][k]):
                    continue
                all_scores[score][k].append(egonet_lp_results[score][k])

        cnt += 1

        print("{:3.2f}% loaded.".format(cnt / num_files * 100), end='\r')

    print()
    # Create directory if not exists
    if not os.path.exists(result_file_base_path + 'cumulated_results'):
        os.makedirs(result_file_base_path + 'cumulated_results')

    # Write data into a single file
    with open(result_file_base_path + "cumulated_results/all-methods-results.pckle", 'wb') as f:
        pickle.dump(all_scores, f, protocol=-1)

    return all_scores


def eval_percent_imp(list, base_score, imp_score, ki):
    base_mean = np.mean(base_score[ki])
    if base_mean != 0:
        list[ki] = (np.mean(imp_score[ki]) - base_mean) / base_mean
    else:
        list[ki] = np.mean(imp_score[ki])


def eval_2_std(base_score, imp_score, ki):
    base_std = 2 * np.std(base_score[ki]) / np.sqrt(len(base_score[ki]))
    imp_std = 2 * np.std(imp_score[ki]) / np.sqrt(len(imp_score[ki]))

    if base_std != 0:
        return ((imp_std - base_std) / base_std) * 100
    else:
        return imp_std * 100


def plot_percent_improvements(result_file_base_path, plot_save_path, comparison_pairs, gather_individual_results=False,
                              facebook_pymk_type=''):
    # comparison_pair should be an array of tuples like: [(base, improved), ('cn', 'dccn'), ('aa', 'dcaa')]

    percent_imp = {}

    for k in top_k_values:
        percent_imp[k] = 0

    # loading result data
    all_scores = get_all_results(result_file_base_path, gather_individual_results)

    for base_method, improved_method in comparison_pairs:

        for k in top_k_values:
            eval_percent_imp(percent_imp, all_scores[base_method], all_scores[improved_method], k)

        imp_mse = {
            'imp': [],
            'imp_err': [],
        }

        for k in top_k_values:
            imp_mse['imp'].append(percent_imp[k] * 100)
            imp_mse['imp_err'].append(eval_2_std(all_scores[base_method], all_scores[improved_method], k))

        plt.figure()
        plt.rc('xtick', labelsize=17)
        plt.rc('ytick', labelsize=17)

        plt.errorbar(top_k_values, imp_mse['imp'], yerr=imp_mse['imp_err'], marker='o', color='b', ecolor='r',
                     elinewidth=2)

        plt.ylabel('Percent Improvement', fontsize=22)
        plt.xlabel('Top K Value', fontsize=22)
        plt.tight_layout()
        current_fig = plt.gcf()
        plt.yticks(np.arange(0, max(imp_mse['imp'])+1, 0.5))
        # plt.yticks(np.arange(0, max(imp_mse['imp_aa']) + 0.5, 0.2))

        plot_name = '{}-vs-{}-improvement'.format(base_method, improved_method)

        if facebook_pymk_type != '':
            plot_name = plot_name + '-' + facebook_pymk_type

        current_fig.savefig('{}{}.pdf'.format(plot_save_path, plot_name), format='pdf')
        plt.clf()

    print("Plotting is Done!")


def get_conf(lp_scores):
    m = np.mean(lp_scores) * 100
    err = 100 * np.std(lp_scores) / np.sqrt(len(lp_scores))

    # up = round(m + err, 2)
    # down = round(m - err, 2)
    # return "({0}, {1})".format(down, up)
    return "{0},{1}".format(round(m, 4), round(err, 4))


def calculate_lp_performance(lp_results_base_file_path, scores=None, is_test=False, specific_triads_only=False,
                             gather_individual_results=False, is_fb=False):
    # Scores is a list of scores to be evaluated ['aa', 'dccn']. If None, all will be evaluated.

    if specific_triads_only:
        lp_results_base_file_path = lp_results_base_file_path + 'test-methods/specific-triads/' + \
                                    scores[0] + '/pickle-files/'
    elif is_test:
        lp_results_base_file_path = lp_results_base_file_path + 'test-methods/' + scores[0] + '/pickle-files/'

    if not is_test and not is_fb and scores is None:
        scores = score_list

    # loading result data
    all_scores = get_all_results(lp_results_base_file_path, gather_individual_results, scores=scores)

    print("Number of egonets analyzed: {0}".format(len(all_scores[scores[0]][top_k_values[0]])))
    print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")

    for score in scores:
        score_name = score
        if score in score_names.keys():
            score_name = score_names[score]

        print(score_name, end=',')

        for k in top_k_values:
            print(get_conf(all_scores[score][k]), end=',')
        print("")


def calculate_lp_performance_on_personalized_triads(lp_results_base_file_path, test_name, gather_individual_results=False):
    # Scores is a list of scores to be evaluated ['aa', 'dccn']. If None, all will be evaluated.

    lp_results_base_file_path = lp_results_base_file_path + 'test-methods/specific-triads/personalized/' + \
                                test_name + '/pickle-files/'

    scores = ['cn', 'aa', 'car', 'cclp']

    # loading result data
    all_scores = get_all_results(lp_results_base_file_path, gather_individual_results, scores=scores)

    print("Number of egonets analyzed: {0}".format(len(all_scores[scores[0]][top_k_values[0]])))
    print("K:,\t 1,\t 3,\t 5,\t 10,\t 15,\t 20,\t 25,\t 30")

    for score in scores:
        score_name = score
        if score in score_names.keys():
            score_name = score_names[score]

        print(score_name, end=',')

        for k in top_k_values:
            print(get_conf(all_scores[score][k]), end=',')
        print("")