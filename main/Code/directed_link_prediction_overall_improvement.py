import os
import pickle
import numpy as np


def calc_percent_imp(percent_imp_list, results, base_score, improved_score, ki):
    if results[base_score][ki] != 0:
        percent_imp_list.append((results[improved_score][ki] - results[base_score][ki]) / results[base_score][ki])
    else:
        percent_imp_list.append(results[improved_score][ki])


# LP results pickle files are in the following order
#   cn, dccn_i, dccn_o, aa_i, aa_o, dcaa_i, dcaa_o


score_list = ['cn', 'dccn_i', 'dccn_o', 'aa_i', 'aa_o', 'dcaa_i', 'dcaa_o']
triangle_types = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09']
result_file_base_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/pickle-files/'
plot_save_path = '/shared/Results/EgocentricLinkPrediction/main/lp/gplus/plots'
top_k_values = [20, 25, 30]

percent_imp_cn_i = []
percent_imp_cn_o = []
percent_imp_aa_i = []
percent_imp_aa_o = []

# # Computing percent improvements and standard errors over all the ego nets analyzed
for triangle_type in triangle_types:
    # loading result data
    for result_file in os.listdir(result_file_base_path + triangle_type):
        with open(result_file_base_path + triangle_type + '/' + result_file, 'rb') as f:
            egonet_lp_results = pickle.load(f)

        for k in top_k_values:
            calc_percent_imp(percent_imp_cn_i, egonet_lp_results, 'cn', 'dccn_i', k)
            calc_percent_imp(percent_imp_cn_o, egonet_lp_results, 'cn', 'dccn_o', k)
            calc_percent_imp(percent_imp_aa_i, egonet_lp_results, 'aa_i', 'dcaa_i', k)
            calc_percent_imp(percent_imp_aa_o, egonet_lp_results, 'aa_o', 'dcaa_o', k)

    print(triangle_type + ": Done")

m_cn_i = np.mean(percent_imp_cn_i)
m_cn_o = np.mean(percent_imp_cn_o)
m_aa_i = np.mean(percent_imp_aa_i)
m_aa_o = np.mean(percent_imp_aa_o)

print("Overall improvement CN in-degree: {0:.4f}%".format(m_cn_i * 100))
print("Overall improvement CN out-degree: {0:.4f}%".format(m_cn_o * 100))
print("Overall improvement AA in-degree: {0:.4f}%".format(m_aa_i * 100))
print("Overall improvement AA out-degree: {0:.4f}%".format(m_aa_o * 100))
print("Overall improvement CN: {0:.4f}%".format(100 * (m_cn_i + m_cn_o) / 2))
print("Overall improvement AA: {0:.4f}%".format(100 * (m_aa_i + m_aa_o) / 2))
