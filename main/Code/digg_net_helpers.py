import re
import math
import networkx as nx
import matplotlib.pyplot as plt

digg_friends_file_path = '/shared/DataSets/Digg2009/raw/digg_friends.csv'


def read_graph():
    print("Reading the original Digg graph...")

    original_graph = nx.Graph()

    with open(digg_friends_file_path, 'r') as f:
        for line in f:
            line = line.rstrip().replace('"', '').split(',')

            # only use mutual friendships and with a valid timestamp
            if line[0] == "0" or line[1] == "0":
                continue

            original_graph.add_edge(int(line[2]), int(line[3]), timestamp=int(line[1]))

    print("Digg graph in.")
    return original_graph


def get_first_and_last_timestamps(graph):
    edges = list(graph.edges(data=True))

    first_timestamp = last_timestamp = edges[0][2]['timestamp']

    for u, v, e in edges:
        if e['timestamp'] < first_timestamp:
            first_timestamp = e['timestamp']

        if e['timestamp'] > last_timestamp:
            last_timestamp = e['timestamp']

    return first_timestamp, last_timestamp


def divide_to_snapshots(graph, length_of_snapshots_in_days):
    orig_snapshots = []

    first_timestamp, last_timestamp = get_first_and_last_timestamps(graph)
    timestamp_duration = length_of_snapshots_in_days * 24 * 60 * 60

    num_of_snapshots = int(math.floor((last_timestamp - first_timestamp) / timestamp_duration))

    for i in range(1, num_of_snapshots + 1):
        orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in graph.edges(data=True)
                                        if d['timestamp'] <= (first_timestamp + timestamp_duration * i)]))

    # check if any edge is left based on the snapshot numbers
    if (last_timestamp - first_timestamp) % timestamp_duration != 0:
        orig_snapshots.append(graph)

    print("Snapshots extracted!")
    return orig_snapshots

# graph = read_graph()
# snapshots = divide_to_snapshots(graph, 90)
#
# print("num of snapshots: {0}".format(len(snapshots)))
# last_num_edges = 0
# last_num_nodes = 0
#
# count = 1
# for snapshot in snapshots:
#     print("Snapshot Number: {0}".format(count), end='')
#     temp = snapshot.number_of_edges()
#     print("Number of edges: {0} - Added: {1}".format(temp, temp - last_num_edges), end='')
#     last_num_edges = temp
#     temp = snapshot.number_of_nodes()
#     print("Number of nodes: {0} - Added: {1} \n".format(temp, temp - last_num_nodes))
#     last_num_nodes = temp
#     count += 1
#
#
# # divide_to_snapshots(graph, 60)
# # divide_to_snapshots(graph, 90)
# # divide_to_snapshots(graph, 180)
#
#
# # list_of_timestamps = []
# # for u, v, t in edges:
# #     list_of_timestamps.append(t)
# #
# # print("hist")
# # print(len(list_of_timestamps))
# #
# # n, bins, patches = plt.hist(list_of_timestamps, 50, density=True, facecolor='g', alpha=0.75)
# #
# # print("show")
# #
# # plt.xlabel('Timestamps')
# # plt.ylabel('Probability')
# # plt.title('Histogram of Friendship timestamps')
# # plt.show()