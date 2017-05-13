import re
import igraph


# Reading facebook data
def read_facebook_graph():
    file = open("../Data/facebook-links.txt", 'r')

    print("Reading the original graph...")

    #original_graph = nx.Graph()
    original_graph = igraph.Graph()

    original_graph.vs['name'] = []
    original_graph.es['timestamp'] = []

    for l in file:
        p = re.compile('\d+')
        nums = p.findall(l)

        if not original_graph.vs(name=nums[0]):
            original_graph.add_vertex(nums[0])

        if not original_graph.vs(name=nums[1]):
            original_graph.add_vertex(nums[1])

        if len(nums) == 2:
            nums.append(-1)
        else:
            nums[2] = int(nums[2])

        original_graph.add_edge(nums[0], nums[1], timestamp=nums[2])

    print("Original graph in.")
    print(original_graph)
    return original_graph

read_facebook_graph()