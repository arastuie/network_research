import networkx as nx

import link_prediction_helpers as helpers

# loading the graph
graph = nx.read_gml("polblogs.gml")
graph = graph.to_undirected()
# graph = helpers.read_facebook_graph('facebook-links.txt')

print("Number of nodes: %s" % nx.number_of_nodes(graph))
print("Number of edges: %s" % nx.number_of_edges(graph))

ego_graphs = helpers.explode_to_ego_centric(graph)
print("Done extracting ego centric networks")

helpers.plot_num_of_edge_node_hist(ego_graphs)

helpers.plot_cdf(ego_graphs)

counter = 0
lp_results = []

for graph in ego_graphs:
    if nx.number_of_edges(graph) < 10 or len(list(nx.non_edges(graph))) < 1:
        continue

    lp_results.append(helpers.run_link_prediction(graph))

    if counter > 2000:
        break

    counter += 1
    print(counter)

helpers.plot_auroc_hist(lp_results)
helpers.plot_pr_hist(lp_results)