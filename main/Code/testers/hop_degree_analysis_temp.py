import sys
import numpy as np
import networkx as nx
import Code.helpers as h


def get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(current_snapshot, next_snapshot, ego_node):
    current_snap_first_hop_nodes = set(current_snapshot.neighbors(ego_node))

    current_snap_second_hop_nodes = set(current_snapshot.nodes()) - current_snap_first_hop_nodes
    current_snap_second_hop_nodes.remove(ego_node)

    next_snap_first_hop_nodes = set(next_snapshot.neighbors(ego_node))

    formed_edges_nodes_with_second_hop = next_snap_first_hop_nodes.intersection(current_snap_second_hop_nodes)

    not_formed_edges_nodes_with_second_hop = current_snap_second_hop_nodes - formed_edges_nodes_with_second_hop

    return formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes


def run_hop_local_degree_analysis(ego_net_snapshots, ego_node, snap_range, ego_net_num, save_plot=False, plot_save_path=''):
    degree_formed_in_snapshots = []
    degree_not_formed_in_snapshots = []
    num_first_hop_nodes_in_snapshots = []
    # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
    for i in snap_range:
        if nx.degree(ego_net_snapshots[i], ego_node) < 500:
            continue

        formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes = \
            get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(ego_net_snapshots[i],
                                                                                ego_net_snapshots[i + 1], ego_node)

        if len(formed_edges_nodes_with_second_hop) == 0:
            continue

        degree_formed = []
        for u in formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            temp_degree_formed = []

            for c in common_neighbors:
                # first hop test
                temp_degree_formed.append(len(set(ego_net_snapshots[i].neighbors(c))
                                              .intersection(current_snap_first_hop_nodes)))

            degree_formed.append(np.mean(temp_degree_formed))

        degree_not_formed = []
        for u in not_formed_edges_nodes_with_second_hop:
            common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
            temp_degree_not_formed = []

            for c in common_neighbors:
                temp_degree_not_formed.append(len(set(ego_net_snapshots[i].neighbors(c))
                                                  .intersection(current_snap_first_hop_nodes)))

            degree_not_formed.append(np.mean(temp_degree_not_formed))

        if len(degree_formed) != 0 and len(degree_not_formed) != 0:
            degree_formed_in_snapshots.append(degree_formed)
            degree_not_formed_in_snapshots.append(degree_not_formed)

        if len(degree_formed_in_snapshots) != 0:
            if save_plot:
                plot_save_path += '/ego_net_%d_hop_degree.png' % ego_net_num

            h.plot_formed_vs_not_local_degree(degree_formed_in_snapshots, degree_not_formed_in_snapshots,
                                              num_first_hop_nodes_in_snapshots, ego_net_num,
                                              save_plot=save_plot, save_path=plot_save_path)

            print("Graph analyzed! {0}".format(ego_net_num))

        return


# def get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(current_snapshot, next_snapshot, ego_node):
#     current_snap_first_hop_nodes = set(current_snapshot.neighbors(ego_node))
#
#     current_snap_second_hop_nodes = set(current_snapshot.nodes()) - current_snap_first_hop_nodes
#     current_snap_second_hop_nodes.remove(ego_node)
#
#     next_snap_first_hop_nodes = set(next_snapshot.neighbors(ego_node))
#
#     formed_edges_nodes_with_second_hop = next_snap_first_hop_nodes.intersection(current_snap_second_hop_nodes)
#
#     not_formed_edges_nodes_with_second_hop = current_snap_second_hop_nodes - formed_edges_nodes_with_second_hop
#
#     return formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes
#
#
# def run_hop_local_degree_analysis(ego_net_snapshots, ego_node, snap_range):
#     mfem = []
#     mnfem = []
#
#     # only goes up to one to last snap, since it compares every snap with the next one, to find formed edges.
#     for i in snap_range:
#         if nx.degree(ego_net_snapshots[i], ego_node) < 30:
#             continue
#
#         formed_edges_nodes_with_second_hop, not_formed_edges_nodes_with_second_hop, current_snap_first_hop_nodes = \
#             get_nodes_of_formed_and_non_formed_edges_between_ego_and_second_hop(ego_net_snapshots[i],
#                                                                                 ego_net_snapshots[i + 1], ego_node)
#
#         if len(formed_edges_nodes_with_second_hop) == 0:
#             continue
#
#         degree_formed = []
#         for u in formed_edges_nodes_with_second_hop:
#             common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
#             temp_degree_formed = []
#
#             for c in common_neighbors:
#                 # first hop test
#                 temp_degree_formed.append(len(set(ego_net_snapshots[i].neighbors(c))
#                                               .intersection(current_snap_first_hop_nodes)))
#
#             degree_formed.append(np.mean(temp_degree_formed))
#
#         degree_not_formed = []
#         for u in not_formed_edges_nodes_with_second_hop:
#             common_neighbors = nx.common_neighbors(ego_net_snapshots[i], u, ego_node)
#             temp_degree_not_formed = []
#
#             for c in common_neighbors:
#                 temp_degree_not_formed.append(len(set(ego_net_snapshots[i].neighbors(c))
#                                                   .intersection(current_snap_first_hop_nodes)))
#
#             degree_not_formed.append(np.mean(temp_degree_not_formed))
#
#         if len(degree_formed) > 0 and len(degree_not_formed) > 0:
#             snap_first_hop_len = len(current_snap_first_hop_nodes)
#             mfem.append(np.mean(degree_formed) / snap_first_hop_len)
#             mnfem.append(np.mean(degree_not_formed) / snap_first_hop_len)
#
#     if len(mfem) > 0:
#         return np.mean(mfem), np.mean(mnfem)
#
#     return -1, -1
#
