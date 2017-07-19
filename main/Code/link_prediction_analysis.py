import pickle
import helpers
import link_prediction_helpers as lp_helpers

print("Reading in 200 random Facebook ego networks...")

with open('../Data/random_200_ego_nets.pckl', 'rb') as f:
    ego_centric_networks, ego_nodes = pickle.load(f)

print("Networks in!")

for i in range(len(ego_centric_networks)):
    print(lp_helpers.run_adamic_adar_on_ego_net(ego_centric_networks[i], ego_nodes[i]))
