import pickle
import helpers
import link_prediction_helpers as lp_helpers

print("Reading in 200 random Facebook ego networks...")

with open('../Data/random_200_ego_nets.pckl', 'rb') as f:
    ego_centric_networks, ego_nodes = pickle.load(f)

print("Networks in!")

lp_results = []
for i in range(len(ego_centric_networks)):
    scores = lp_helpers.run_adamic_adar_on_ego_net(ego_centric_networks[i], ego_nodes[i])

    if scores is not None:
        lp_results.append(scores)

lp_helpers.plot_auroc_hist(lp_results)
lp_helpers.plot_pr_hist(lp_results)
