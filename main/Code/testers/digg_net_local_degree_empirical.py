import pickle
import numpy as np
import networkx as nx
import digg_helpers as digg

# u -> z -> v

def get_formed_and_not_v_nodes(current_snapshot, next_snapshot, ego_node):
    current_snap_z_nodes = set(current_snapshot.neighbors(ego_node))
    current_snap_v_nodes = set(current_snapshot.nodes()) - current_snap_z_nodes
    current_snap_v_nodes.remove(ego_node)

    next_snap_z_nodes = set(next_snapshot.neighbors(ego_node))

    formed_v_nodes_with_ego = next_snap_z_nodes.intersection(current_snap_v_nodes)
    not_formed_v_nodes_with_ego = current_snap_v_nodes - formed_v_nodes_with_ego

    return formed_v_nodes_with_ego, not_formed_v_nodes_with_ego, current_snap_z_nodes


snapshot_duration_in_days = 90
orig_snaps = digg.divide_to_snapshots(digg.read_graph(), snapshot_duration_in_days)
analyzed_nodes = set()
results = {}

for snap in range(len(orig_snaps) - 1):
    results[snap] = {}

print("Start local degree empirical analysis...")

for snap_index in range(len(orig_snaps) - 1):
    nodes_in_snap = set(orig_snaps[snap_index].nodes())
    print(snap_index)
    for ego in nodes_in_snap:
        if ego in analyzed_nodes:
            continue

        # extract all ego nets for ease of computation
        ego_snaps = []
        for i in range(snap_index, len(orig_snaps)):
            ego_snaps.append(nx.ego_graph(orig_snaps[i], ego, radius=2, center=True))

        for i in range(len(ego_snaps) - 1):
            formed_v_nodes, not_formed_v_nodes, z_nodes = get_formed_and_not_v_nodes(ego_snaps[i], ego_snaps[i + 1],
                                                                                     ego)
            z_local_formed = []
            z_local_not_formed = []
            z_global_formed = []
            z_global_not_formed = []

            num_first_hop_nodes = nx.degree(ego_snaps[i], ego)

            if len(formed_v_nodes) == 0 or len(not_formed_v_nodes) == 0:
                continue

            for v in formed_v_nodes:
                common_neighbors = nx.common_neighbors(ego_snaps[i], v, ego)
                temp_z_local_formed = []
                temp_z_global_formed = []

                for cn in common_neighbors:
                    temp_z_local_formed.append(len(list(nx.common_neighbors(ego_snaps[i], cn, ego))))
                    temp_z_global_formed.append(nx.degree(ego_snaps[i], cn))

                z_local_formed.append(np.mean(temp_z_local_formed))
                z_global_formed.append(np.mean(temp_z_global_formed))

            for v in not_formed_v_nodes:
                common_neighbors = nx.common_neighbors(ego_snaps[i], v, ego)
                temp_z_local_not_formed = []
                temp_z_global_not_formed = []

                for cn in common_neighbors:
                    temp_z_local_not_formed.append(len(list(nx.common_neighbors(ego_snaps[i], cn, ego))))
                    temp_z_global_not_formed.append(nx.degree(ego_snaps[i], cn))

                z_local_not_formed.append(np.mean(temp_z_local_not_formed))
                z_global_not_formed.append(np.mean(temp_z_global_not_formed))

            results[i][ego] = {
                'num_nodes': ego_snaps[i].number_of_nodes(),
                'num_z_nodes': len(z_nodes),
                'local_degrees_formed': z_local_formed,
                'local_degrees_not_formed': z_local_not_formed,
                'global_degrees_formed': z_global_formed,
                'global_degrees_not_formed': z_global_not_formed,
            }

        analyzed_nodes.add(ego)

# save an empty file in analyzed_egonets to know which ones were analyzed
with open('/shared/Results/EgocentricLinkPrediction/main/empirical/digg/pickle-files/{0}-days-duration-results-1.pckle'
          .format(snapshot_duration_in_days), 'wb') as f:
    pickle.dump(results, f, protocol=-1)
