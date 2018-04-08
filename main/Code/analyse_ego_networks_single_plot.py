import os
import pickle
import numpy as np
import hop_degree_analysis_temp as a6

print("Analysing ego centric networks...")
path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/pickle-files/local/lower-6/temp'


def run_analysis(index):
    mfems = []
    mnfems = []

    cnt = 0

    for ego_net_file in os.listdir('/shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}'.format(index)):
        with open('./shared/DataSets/FacebookViswanath2009/egocentric/fb-egonets/{0}/{1}'.format(index, ego_net_file), 'rb') as f:
            egonet_snapshots, ego_node = pickle.load(f)

        mfem, mnfem = a6.run_hop_local_degree_analysis(egonet_snapshots, ego_node, range(0, 5))
        # mfem, mnfem = a8.run_hop_global_degree_analysis(egonet_snapshots, ego_node, 0, False, '../Plots/global_degree')

        if mfem != -1:
            mfems.append(mfem)
            mnfems.append(mnfem)

        cnt = cnt + 1

        if cnt % 500 == 0:
            print("Index {0} -> {1}".format(index, cnt))

    if len(mfems) > 0:
        print("mfem -> {0}".format(np.mean(mfems)))
        print("mnfem -> {0}".format(np.mean(mnfems)))

        with open('{0}/{1}-egonodes-result.pckl'.format(path, index), 'wb') as f:
            pickle.dump([mfems, mnfems], f, protocol=-1)
    else:
        print("No analysis in index {0}".format(index))

run_analysis(0)
