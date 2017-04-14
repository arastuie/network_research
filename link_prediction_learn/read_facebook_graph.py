import networkx as nx
import re

file = open('facebook-links.txt', 'r')

graph = nx.Graph()


for l in file:
    p = re.compile('\d+')
    nums = p.findall(l)

    nums[0] = int(nums[0])
    nums[1] = int(nums[1])

    if not graph.has_node(nums[0]):
        graph.add_node(nums[0])

    if not graph.has_node(nums[1]):
        graph.add_node(nums[1])

    if len(nums) == 2:
        nums.append(-1)
    else:
        nums[2] = int(nums[2])

    graph.add_edge(nums[0], nums[1], timestamp=nums[2])


nx.write_gml(graph, 'facebook_links.gml')

print(graph.number_of_nodes())
print(graph.number_of_edges())