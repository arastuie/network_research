# import pickle
# import helpers
# import link_prediction_helpers as lp_helpers
import linear_regression_link_prediction as lr
import numpy as np
import os

# print("Reading in 200 random Facebook ego networks...")
#
# with open('../Data/random_200_ego_nets.pckl', 'rb') as f:
#     ego_centric_networks, ego_nodes = pickle.load(f)
#
# print("Networks in!")
#
# lp_results = []
# for i in range(len(ego_centric_networks)):
#     scores = lp_helpers.run_adamic_adar_on_ego_net(ego_centric_networks[i], ego_nodes[i])
#
#     if scores is not None:
#         lp_results.append(scores)
#
# lp_helpers.plot_auroc_hist(lp_results)
# lp_helpers.plot_pr_hist(lp_results)

aa_only_scores = []
both_scores = []
cnt = 1
for path in os.listdir('../Data/fb_lp_features_reciprocal'):
    aa_only_scores.append(list(lr.run_linear_regression('../Data/fb_lp_features_reciprocal/{0}'.format(path), True)))
    both_scores.append(list(lr.run_linear_regression('../Data/fb_lp_features_reciprocal/{0}'.format(path), False)))
    print(cnt, end='\r')
    cnt += 1

aa_only_scores = np.array(aa_only_scores)
print("AA Only:")
print("Mean AA Coefficient: {0}".format(np.mean(aa_only_scores[:, 0])))
print("Mean AUROC: {0}".format(np.mean(aa_only_scores[:, 1])))
print("Mean Avg Precision: {0}".format(np.mean(aa_only_scores[:, 2])))

both_scores = np.array(both_scores)
print("\nAA With DCAA:")
print("Mean AA Coefficient: {0}".format(np.mean(both_scores[:, 0])))
print("Mean DCAA Coefficient: {0}".format(np.mean(both_scores[:, 1])))
print("Mean AUROC: {0}".format(np.mean(both_scores[:, 2])))
print("Mean Avg Precision: {0}".format(np.mean(both_scores[:, 3])))

# aa_only_scores = np.array(aa_only_scores)
# print("AA Only:")
# print("Mean AA Coefficient: {0}".format(np.mean(aa_only_scores[:, 0])))
# print("Mean AA Coefficient ** 2: {0}".format(np.mean(aa_only_scores[:, 1])))
# print("Mean AUROC: {0}".format(np.mean(aa_only_scores[:, 2])))
# print("Mean Avg Precision: {0}".format(np.mean(aa_only_scores[:, 3])))
#
# both_scores = np.array(both_scores)
# print("\nAA With DCAA:")
# print("Mean AA Coefficient: {0}".format(np.mean(both_scores[:, 0])))
# print("Mean AA Coefficient ** 2: {0}".format(np.mean(both_scores[:, 2])))
# print("Mean DCAA Coefficient: {0}".format(np.mean(both_scores[:, 1])))
# print("Mean DCAA Coefficient ** 2: {0}".format(np.mean(both_scores[:, 3])))
# print("Mean AA * DCAA Coefficient: {0}".format(np.mean(both_scores[:, 4])))
# print("Mean AUROC: {0}".format(np.mean(both_scores[:, 5])))
# print("Mean Avg Precision: {0}".format(np.mean(both_scores[:, 6])))
