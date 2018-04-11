import re
import networkx as nx

digg_friends_file_path = '/shared/DataSets/Digg2009/raw/digg_friends.csv'


def read_digg_graph():
    print("Reading the original Digg graph...")

    original_graph = nx.Graph()

    with open(digg_friends_file_path, 'r') as f:
        for line in f:
            line = line.rstrip().replace('"', '').split(',')

            # only use mutual friendships and with a valid timestamp
            if line[0] == "0" or line[1] == "0":
                continue

            original_graph.add_edge(line[2], line[3], timestamp=line[1])

    print("Digg graph in.")
    return original_graph


graph = read_digg_graph()
print(graph.number_of_edges())
print(graph.number_of_nodes())