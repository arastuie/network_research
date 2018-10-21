import os
import glob
import math
import time
import pickle
import numpy as np
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import directed_graphs_helpers as dgh
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_recall_curve
from sklearn.metrics import average_precision_score, auc

#    0           1       2   3   4    5        6           7       8       9                  10                 11                   12               13
# egonet-id, v-node-id, CN, AA, CAR, CCLP, LD-undirectd, LD-in, LD-out, snapshot_index, #_nodes_first_hop, #_nodes_second_hop, #_of_edges_in_egonet, formed?
feature_names = ['Egonet Id', 'V Node Id', 'CN', 'AA', 'CAR', 'CCLP', 'LD Undirectd', 'LD In-degree', 'LD Out-degree',
                 'Snapshot Index', '# first hop nodes', '# second hop nodes', '# of edges', 'Formed?']


def evaluate_lp_measures(ego_net_file, data_file_base_path, result_file_base_path, skip_over_100k=True):
    # egonet-id, v-node-id, CN, AA, CAR, CCLP, LD-undirectd, LD-in, LD-out, snapshot_index, #_nodes_first_hop,
    # #_nodes_second_hop, #_of_edges_in_egonet, formed?
    start_time = datetime.now()

    # return if the egonet is on the analyzed list
    if os.path.isfile(result_file_base_path + 'analyzed_egonets/' + ego_net_file):
        return

    # return if the egonet is on the skipped list
    if skip_over_100k and os.path.isfile(result_file_base_path + 'skipped_egonets/' + ego_net_file):
        return

    with open(data_file_base_path + ego_net_file, 'rb') as f:
        ego_node, ego_net_snapshots = pickle.load(f)

    num_nodes = nx.number_of_nodes(ego_net_snapshots[-1])
    # if the number of nodes in the last snapshot of the network is big, skip it and save a file in skipped-nets
    if skip_over_100k and num_nodes >= 100000:
        with open(result_file_base_path + 'skipped_egonets/' + ego_net_file, 'wb') as f:
            pickle.dump(0, f, protocol=-1)
        return

    gathered_data = 0
    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in range(len(ego_net_snapshots) - 1):
        first_hop_nodes, _, v_nodes = dgh.get_combined_type_nodes(ego_net_snapshots[i], ego_node)

        v_nodes_list = list(v_nodes.keys())

        snapshot_data = np.zeros((len(v_nodes_list), 14))
        snapshot_data[:, 0] = ego_node
        snapshot_data[:, 9] = i
        snapshot_data[:, 10] = len(first_hop_nodes)
        snapshot_data[:, 11] = len(v_nodes_list)
        snapshot_data[:, 12] = ego_net_snapshots[i].number_of_edges()

        for v_i in range(0, len(v_nodes_list)):
            snapshot_data[v_i, 1] = v_nodes_list[v_i]
            snapshot_data[v_i, -1] = 1 if ego_net_snapshots[i + 1].has_edge(ego_node, v_nodes_list[v_i]) else 0

        # a dict of info on z nodes. Every key points to a list [Local_Degree, Local_indegre, local_outdegree, aa, cclp]
        z_info = {}
        # undirected network for CCLP and CAR
        undirected_ego_net = ego_net_snapshots[i].to_undirected()
        for z in first_hop_nodes:
            z_successors = set(ego_net_snapshots[i].successors(z))
            z_predecessors = set(ego_net_snapshots[i].predecessors(z))
            z_neighbors = z_successors.union(z_predecessors)

            # This should be the intersection of z_neighbors with the union of nodes in first and second hops
            z_global_degree = len(z_neighbors)

            z_local_in_degree = len(z_predecessors.intersection(first_hop_nodes))
            z_local_out_degree = len(z_successors.intersection(first_hop_nodes))
            z_local_degree = len(z_neighbors.intersection(first_hop_nodes))

            y = z_global_degree - z_local_degree

            # if y = 1, then the z node has no neighbor in the second hop, thus no need to compute
            if y == 1:
                continue

            z_info[z] = []
            # ld
            z_info[z].append(math.log(z_local_degree + 1))
            # lid
            z_info[z].append(math.log(z_local_in_degree + 1))
            # lod
            z_info[z].append(math.log(z_local_out_degree + 1))
            # aa
            z_info[z].append(1 / math.log(z_global_degree))
            # cclp
            z_info[z].append(nx.triangles(undirected_ego_net, z) / (z_global_degree * (z_global_degree - 1) / 2))

        for v_i in range(len(v_nodes_list)):
            num_cn = len(v_nodes[v_nodes_list[v_i]])

            # cn score
            snapshot_data[v_i, 2] = num_cn

            # car score
            lcl = 0
            if num_cn > 1:
                for zz_i in range(num_cn - 1):
                    for zzz_i in range(zz_i + 1, num_cn):
                        v_z_list = v_nodes[v_nodes_list[v_i]]
                        if ego_net_snapshots[i].has_edge(v_z_list[zz_i], v_z_list[zzz_i]) or \
                                ego_net_snapshots[i].has_edge(v_z_list[zzz_i], v_z_list[zz_i]):
                            lcl += 1

            snapshot_data[v_i, 4] = num_cn * lcl

            temp_ld = 0
            temp_lid = 0
            temp_lod = 0
            temp_aa = 0
            temp_cclp = 0

            for z in v_nodes[v_nodes_list[v_i]]:
                temp_ld += z_info[z][0]
                temp_lid += z_info[z][1]
                temp_lod += z_info[z][2]
                temp_aa += z_info[z][3]
                temp_cclp += z_info[z][4]

            snapshot_data[v_i, 3] = temp_aa
            snapshot_data[v_i, 5] = temp_cclp
            snapshot_data[v_i, 6] = temp_ld
            snapshot_data[v_i, 7] = temp_lid
            snapshot_data[v_i, 8] = temp_aa

        if type(gathered_data) == int:
            gathered_data = snapshot_data
        else:
            gathered_data = np.concatenate((gathered_data, snapshot_data), axis=0)


    # Saving the entire data in a csv
    print("fix this")
    # np.savetxt("{}results/{}.txt".format(result_file_base_path, ego_node), gathered_data, delimiter=",")

    print("Analyzed ego net: {0} - Duration: {1} - Num nodes: {2} - Formed: {3}"
          .format(ego_net_file, datetime.now() - start_time, num_nodes, np.sum(gathered_data[:, -1])))

    # save an empty file in analyzed_egonets to know which ones were analyzed
    # with open(result_file_base_path + 'analyzed_egonets/' + ego_net_file, 'wb') as f:
    #     pickle.dump(0, f, protocol=-1)

    return


def combine_all_egonet_data(result_base_path, num_rows=0, num_samples=0):
    combined_result_path = result_base_path + 'combined/'
    result_base_path = result_base_path + 'results/'

    egonet_list = list(os.listdir(result_base_path))

    if num_samples != 0:
        egonet_list = np.random.choice(egonet_list, size=num_samples, replace=False)

    num_egonets = len(egonet_list)

    print("Number of data samples selected: {}".format(num_egonets))

    # count number of rows if not passed
    cnt = 0
    if num_rows == 0:
        print("Count the total number of rows.")
        for egonet_data_file_name in egonet_list:
            egonet_data = np.load(result_base_path + egonet_data_file_name)
            num_rows += np.shape(egonet_data)[0]
            print("{:3.2f}%".format(100 * cnt / num_egonets), end='\r')
            cnt += 1
        print()

    # combine all sets
    cnt = 0
    nxt_row_start_index = 0
    dataset = np.zeros((num_rows, 14))
    print("Start combining the egonet files...")
    for egonet_data_file_name in egonet_list:
        egonet_data = np.load(result_base_path + egonet_data_file_name)

        data_length = np.shape(egonet_data)[0]

        dataset[nxt_row_start_index:nxt_row_start_index + data_length, :] = egonet_data
        nxt_row_start_index = nxt_row_start_index + data_length

        print("{:3.2f}%".format(100 * cnt / num_egonets), end='\r')
        cnt += 1

    print()
    if not os.path.exists(combined_result_path):
        os.makedirs(combined_result_path)

    if num_samples == 0:
        np.save(combined_result_path + "all.npy", dataset)
    else:
        np.save("{}random_{}k.npy".format(combined_result_path, int(num_samples / 1000)), dataset)
    print("Done.")
    return


def split_data_based_on_snapshot(result_base_path, combined_file_name='all.npy'):
    # this method splits the data into snapshot 1, snapshot 2, and rest of them
    combined_result_path = result_base_path + 'combined/'
    if not os.path.exists(combined_result_path + combined_file_name):
        print("Combined results not found!")
        return

    print("Loading the dataset...")
    dataset = np.load(combined_result_path + combined_file_name)
    print("Dataset loaded.")

    # snapshot 0
    split_dataset = dataset[np.where(dataset[:, 9] == 0)[0], :]
    np.save("{}snapshot-{}-{}".format(combined_result_path, 0, combined_file_name), split_dataset)
    del split_dataset
    print("Snapshot 0 done.")

    # snapshot 1
    split_dataset = dataset[np.where(dataset[:, 9] == 1)[0], :]
    np.save("{}snapshot-{}-{}".format(combined_result_path, 1, combined_file_name), split_dataset)
    del split_dataset
    print("Snapshot 1 done.")

    # snapshots > 1
    split_dataset = dataset[np.where(dataset[:, 9] > 1)[0], :]
    np.save("{}snapshots-after-1-{}".format(combined_result_path, combined_file_name), split_dataset)
    del split_dataset
    print("Snapshots after 1 done.")

    return


def preprocessing(result_base_path, data_set_file_name, feature_indices, for_testing=False):
    combined_result_path = result_base_path + 'combined/'
    print("Loading the dataset: {}combined/{}".format(result_base_path, data_set_file_name))
    dataset = np.load(combined_result_path + data_set_file_name)
    print("Dataset loaded.")

    np.random.shuffle(dataset)

    y = dataset[:, -1].astype(int)
    # x = dataset[:, [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]]
    # x = dataset[:, [2, 3, 4, 5, 10, 11, 12]]
    x = dataset[:, feature_indices]

    if not for_testing:
        del dataset
        return x, y

    x_ego_snap = dataset[:, [0, 9]]
    return x, y, x_ego_snap


def train_random_forest(result_base_path, train_set_file_name, model_name, feature_indices, n_jobs=6,
                        test_set_file_name=''):
    x_train, y_train = preprocessing(result_base_path, train_set_file_name, feature_indices)

    # Training the classifier
    print("\nStart model fitting with {} samples. {:2.3f}% positive examples.".format(len(y_train), 100 * sum(y_train)
                                                                                      / len(y_train)))

    clf = RandomForestClassifier(n_estimators=300, max_features='sqrt', max_depth=80, n_jobs=n_jobs, criterion='gini')

    start = time.time()
    clf = clf.fit(x_train, y_train)
    end = time.time()
    print("Training took {:10.3f} min".format((end - start) / 60))

    print_rf_feature_importance(clf, feature_indices)

    if not os.path.exists(result_base_path + 'trained_models'):
        os.makedirs(result_base_path + 'trained_models')

    trained_model_file_name = "RF-{}.pickle".format(model_name)
    with open("{}trained_models/{}".format(result_base_path, trained_model_file_name), 'wb') as f:
        pickle.dump(clf, f, protocol=-1)

    if test_set_file_name != '':
        test_trained_model(result_base_path, test_set_file_name, trained_model_file_name, feature_indices)

    return


def test_trained_model(result_base_path, test_set_file_name, trained_model_file_name, feature_indices,
                       predictions_file_name):
    x_test, y_test, x_ego_snap = preprocessing(result_base_path, test_set_file_name, feature_indices, for_testing=True)

    with open("{}trained_models/{}".format(result_base_path, trained_model_file_name), 'rb') as f:
        clf = pickle.load(f)

    print_rf_feature_importance(clf, feature_indices)

    # predicting
    print("\nStart predicting {} samples. {:2.3f}% positive examples.".format(len(y_test), 100 * sum(y_test)
                                                                              / len(y_test)))
    start = time.time()
    y_pred = clf.predict_proba(x_test)[:, 1]
    # y_pred = clf.predict(x_test)
    end = time.time()
    print("Prediction took {:10.3f} min".format((end - start) / 60))

    # putting all data together. egonet_id, snapshot_id, y_test, y_pred
    predicted_data = np.concatenate((x_ego_snap, np.reshape(y_test, (len(y_test), 1)),
                                     np.reshape(y_pred, (len(y_pred), 1))), axis=1)

    np.save("{}predictions/{}".format(result_base_path, predictions_file_name), predicted_data)

    # precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    # print("precision: {}, recall: {}, f1_score: {}".format(precision, recall, f1_score))
    #
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(cnf_matrix)

    return


def calc_percision_at_k(result_base_path, predicted_data_file_path, k_values=[1, 3, 5, 10, 15, 20, 25, 30]):
    predicted_data = np.load("{}predictions/{}".format(result_base_path, predicted_data_file_path))

    # predicted_data must be in the following format: egonet_id, snapshot_id, y_test, y_pred
    unique_egos = np.unique(predicted_data[:, 0])

    top_k_res = {}
    for k in k_values:
        top_k_res[k] = []

    for ego in unique_egos:
        temp_top_k_res = {}
        for k in k_values:
            temp_top_k_res[k] = []
        ego_data_indices = np.where(predicted_data[:, 0] == ego, True, False)
        ego_data = predicted_data[ego_data_indices, :]
        predicted_data = predicted_data[~ego_data_indices, :]
        unique_snapshots = np.unique(ego_data[:, 1])
        for snap in unique_snapshots:
            ego_snap_data = ego_data[np.where(ego_data[:, 1] == snap)[0], :]
            res = ego_snap_data[np.argsort(-ego_snap_data[:, 3]), 2]

            for k in k_values:
                if len(res) >= k:
                    temp_top_k_res[k].append(sum(res[:k]) / k)

        for k in k_values:
            if len(temp_top_k_res[k]) > 0:
                top_k_res[k].append(np.mean(temp_top_k_res[k]))

    for k in k_values:
        top_k_res[k] = np.mean(top_k_res[k])

    print(top_k_res)
    return top_k_res


def aupr_trained_model(result_base_path, test_set_file_name, trained_model_file_name, feature_indices, plot_name,
                       get_feature_importance=False):
    x_test, y_test = preprocessing(result_base_path + 'pickle-files/', test_set_file_name, feature_indices)

    with open("{}trained_models/{}".format(result_base_path + 'pickle-files/', trained_model_file_name), 'rb') as f:
        clf = pickle.load(f)

    if get_feature_importance:
        print_rf_feature_importance(clf, feature_indices)

    # predicting
    print("\nStart predicting {} samples. {:2.3f}% positive examples.".format(len(y_test), 100 * sum(y_test)
                                                                              / len(y_test)))
    start = time.time()
    y_pred = clf.predict_proba(x_test)[:, 1]
    end = time.time()
    print("Prediction took {:10.3f} min".format((end - start) / 60))

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)
    aupr = auc(precision, recall, reorder=True)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    plt.title('{} Precision-Recall curve: AP={:0.4f}, AUPR={:0.4f}'.format(plot_name, average_precision,
                                                                           aupr))
    current_fig = plt.gcf()
    current_fig.savefig("{}plots/{}-aupr.pdf".format(result_base_path, plot_name), format='pdf')
    plt.clf()

    return


def print_rf_feature_importance(clf, feature_indices):
    feature_importance = clf.feature_importances_
    print("\tFeature Importance:\n\t", end='')
    for i in range(len(feature_importance)):
        print("{}: {:1.4f}".format(feature_names[feature_indices[i]], feature_importance[i]), end=' - ')

    return


############ Roller Learner ##############
def preprocessing_5_to_1_train(train_set, feature_indices):
    print("\tStart preprocessing...")
    # down sample negative examples, 1 positive to 5 negative
    num_positive_ex = np.sum(train_set[:, 13])
    # Number of negative examples to del is the total number of examples - pos_ex - pos_ex * 5 which is 1 - 5 ratio
    num_negatives_to_del = int(np.shape(train_set)[0] - num_positive_ex * 6)

    # This if is not going to happen, so let everything fail, is it does.
    if num_negatives_to_del < 0:
        print("\tThere are not enough negative examples to down-sample.")

    neg_indices_to_del = np.random.choice(np.where(train_set[:, 13] == 0)[0], size=num_negatives_to_del, replace=False)
    train_set = np.delete(train_set, neg_indices_to_del, 0)
    print("\tDown-sampling done.")

    np.random.shuffle(train_set)

    y = train_set[:, -1].astype(int)
    x = train_set[:, feature_indices]

    return x, y


def train_rf_roller_learner(train_set, feature_indices, n_jobs, trained_model_file_path):
    x_train, y_train = preprocessing_5_to_1_train(train_set, feature_indices)

    # Training the classifier
    print("\tStart model fitting with {} samples.".format(len(y_train)))

    clf = RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=80, n_jobs=n_jobs, criterion='gini')

    start = time.time()
    clf = clf.fit(x_train, y_train)
    end = time.time()
    print("\tTraining took {:10.3f} min".format((end - start) / 60))

    with open(trained_model_file_path, 'wb') as f:
        pickle.dump(clf, f, protocol=-1)

    print_rf_feature_importance(clf, feature_indices)

    return clf


def test_roller_learner(test_set, clf, feature_indices, pred_save_path):
    np.random.shuffle(test_set)

    y_test = test_set[:, -1].astype(int)
    x_test = test_set[:, feature_indices]
    x_egos = test_set[:, 0]

    # predicting
    print("\tStart predicting {} samples.".format(len(y_test)))

    start = time.time()
    y_pred = clf.predict_proba(x_test)[:, 1]
    end = time.time()
    print("\nPrediction took {:10.3f} min".format((end - start) / 60))

    # putting all data together. egonet_id, y_test, y_pred
    predicted_data = np.stack((x_egos, y_test, y_pred), axis=-1)
    np.save(pred_save_path, predicted_data)

    return predicted_data


def roller_percision_at_k(predicted_data, k_values):
    print("\tStart calculating P@K...")
    start = time.time()
    # predicted_data must be in the following format: egonet_id, y_test, y_pred
    unique_egos = np.unique(predicted_data[:, 0])

    top_k_res = {}
    for k in k_values:
        top_k_res[k] = []

    for ego in unique_egos:
        ego_data_indices = np.where(predicted_data[:, 0] == ego, True, False)
        ego_data = predicted_data[ego_data_indices, :]
        predicted_data = predicted_data[~ego_data_indices, :]

        res = ego_data[np.argsort(-ego_data[:, 2]), 1]

        for k in k_values:
            if len(res) >= k:
                top_k_res[k].append(sum(res[:k]) / k)

    for k in k_values:
        top_k_res[k] = np.mean(top_k_res[k])

    end = time.time()
    print("\tP@K calc took {:10.3f} min".format((end - start) / 60))

    return top_k_res


def random_forest_roller_learner(result_base_path, data_file_name, model_name, feature_indices, n_jobs=6):
    k_values = [1, 3, 5, 10, 15, 20, 25, 30]

    result_path = result_base_path + 'trained_models/' + model_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    top_k_res = {}
    for k in k_values:
        top_k_res[k] = []

    print("Loading the dataset: {}combined/{}".format(result_base_path, data_file_name))
    dataset = np.load(result_base_path + 'combined/' + data_file_name)
    print("Dataset loaded.")

    # get the number of distinct snapshots
    last_snap_index = int(dataset[:, 9].max())
    print("{} snapshots to be evaluated.".format(last_snap_index))

    # goes up to last snap - 1
    for sid in range(last_snap_index):
        trained_model_file_path = "{}/clf-sid-{}.pickle".format(result_path, sid)
        test_res_file_path = '{}/preds-sid-{}.npy'.format(result_path, sid + 1)

        # Training on this snapshot
        if os.path.isfile(trained_model_file_path):
            print("\n\n\nLoading classifier on snapshot {}".format(sid))
            with open(trained_model_file_path, 'rb') as f:
                clf = pickle.load(f)
        else:
            print("\n\n\nStart training on snapshot {}".format(sid))
            train_set = dataset[np.where(dataset[:, 9] == sid)[0], :]
            clf = train_rf_roller_learner(train_set, feature_indices, n_jobs, trained_model_file_path)

        # Testing on the next snapshot
        if os.path.isfile(test_res_file_path):
            print("\tLoading test result on snapshot {}".format(sid + 1))
            predicted_data = np.load(test_res_file_path)
        else:
            test_set = dataset[np.where(dataset[:, 9] == sid + 1)[0], :]
            predicted_data = test_roller_learner(test_set, clf, feature_indices, test_res_file_path)

        # Calculating P@K
        top_k_res_temp = roller_percision_at_k(predicted_data, k_values)
        print(top_k_res)

        for k in k_values:
            top_k_res[k].append(top_k_res_temp[k])

    # Printing final results
    print(top_k_res)
    for k in k_values:
        top_k_res[k] = np.mean(top_k_res[k])

    print(top_k_res)

    return
