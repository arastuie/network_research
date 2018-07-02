import time
import pickle
import networkx as nx


dataset_file_path = '/shared/DataSets/FacebookViswanath2009/raw/facebook-links.txt'
egonet_files_path = '/shared/DataSets/FacebookViswanath2009/egocentric/all_egonets/'


# Reading facebook data
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
