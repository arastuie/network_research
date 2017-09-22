import networkx as nx
import helpers as helpers
import math
import os
from joblib import Parallel, delayed


def get_ego_net(ego_node, orig_snaps):
    ego_snapshots = []
    for i in range(len(orig_snaps)):
        ego_snapshots.append(nx.ego_graph(orig_snaps[i], ego_node, radius=2, center=True))

    if len(ego_snapshots[0].neighbors(ego_node)) < 50:
        return

    total_edges_formed = 0

    file = open('../Data/fb-lp-features-degree-vs-local-degree/{0}.txt'.format(ego_node), 'w')

    for s in range(len(ego_snapshots) - 1):
        first_hop_nodes = set(ego_snapshots[s].neighbors(ego_node))

        second_hop_nodes = set(ego_snapshots[s].nodes()) - first_hop_nodes
        second_hop_nodes.remove(ego_node)

        density = nx.density(ego_snapshots[s])
        average_num_edge = nx.number_of_edges(ego_snapshots[s]) / (nx.number_of_nodes(ego_snapshots[s]) / 2)

        for n in second_hop_nodes:
            first_hop_degrees = []
            total_degrees = []
            common_neighbors = nx.common_neighbors(ego_snapshots[s], ego_node, n)

            for c in common_neighbors:
                cn_neighbors = set(nx.neighbors(ego_snapshots[s], c))
                first_hop_degrees.append(len(cn_neighbors.intersection(first_hop_nodes)))
                total_degrees.append(len(cn_neighbors))

            # for ii in range(len(first_hop_degrees)):
            #     if first_hop_degrees[ii] == 0:
            #         first_hop_degrees[ii] = 1.5
            #     elif first_hop_degrees[ii] == 1:
            #         first_hop_degrees[ii] = 1.75
            #
            # aa_index = sum(1 / math.log(d) for d in total_degrees)
            # dc_aa_index = sum(1 / math.log(d) for d in first_hop_degrees)

            did_form = ego_snapshots[s + 1].has_edge(ego_node, n)
            file.write("{0},{1},{2},{3},{4},{5}\n".format(aa_index, dc_aa_index, density, average_num_edge, s,
                                                          int(did_form)))

            if did_form:
                total_edges_formed += 1

    file.close()

    if total_edges_formed < 10:
        os.remove('../Data/fb_lp_features_reciprocal/{0}.txt'.format(ego_node))
    else:
        print("Network in.")


fb_graph = helpers.read_facebook_graph()

orig_snapshots = []

oldest_timestamp = 1157454929
seconds_in_90_days = 7776000
for i in range(10):
    orig_snapshots.append(nx.Graph([(u, v, d) for u, v, d in fb_graph.edges(data=True)
                                    if d['timestamp'] < (oldest_timestamp + seconds_in_90_days * i)]))
orig_snapshots.append(fb_graph)
fb_graph = None

print("Start analyzing...")
Parallel(n_jobs=20)(delayed(get_ego_net)(node, orig_snapshots[:]) for node in orig_snapshots[0].nodes())
