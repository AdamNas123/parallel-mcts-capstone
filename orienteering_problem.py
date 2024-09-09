import math
from typing import Tuple
# Dictionary of dictionaries
# Key: Node index
# Value: Dictionary of neighbours with corresponding distances from node

"""
Node class represents each node in the Orienteering Problem Dataset
Contains:
  -  index
  -  x and y pos
  -  value (reward associated with travelling to node)
  - dict of neighbours (neighbour_index : distance from neighbour)
"""
class Node:
    def __init__(self, index, x, y, value):
        self.index = index
        self.x = x
        self.y = y
        self.value = value
        self.neighbours = {}
    
    def add_neighbour(self, neighbour, distance):
        self.neighbours[neighbour] = distance


"""
Graph represents the entire connection of nodes in the Orienteering Problem Dataset.
Contains:
  -  graph:
     -  Dictionary of node indexes to node objects (node_index : Node)
  -  budget (Travel budget defined in dataset - Distance travelled contributes to budget) 
  -  
"""
class OrienteeringGraph:
    def __init__(self, budget):
        self.graph = {}
        self.budget = budget

    def add_node(self, index, x, y, value):
        if index not in self.graph:
            self.graph[index] = Node(index, x, y, value)

    def add_edge(self, index1, index2, distance):
        self.graph[index1].add_neighbour(index2, distance)
        self.graph[index2].add_neighbour(index1, distance)

    def get_node(self, index):
        return self.graph.get(index)
    
    def get_nodes(self):
        return list(self.graph.keys())

    def get_neighbours(self, index):
        return self.graph[index].neighbours if index in self.graph else None
    
    def get_distance(self, node1, node2):
        if node1 in self.graph and node2 in self.graph[node1].neighbours:
            return self.graph[node1].neighbours[node2]
        else:
            return None

    def print_graph(self):
        for node_index, node in self.graph.items():
            print(f"Node {node_index}: X: {node.x}, Y: {node.y}, Value: {node.value}")
            for neighbour_index, distance in node.neighbours.items():
                print(f"  -> Connected to Node {neighbour_index} with distance {distance:.2f}")
            print()  # Add an empty line for better readability
            print(f"Problem with budget {self.budget}")
   
def calculate_euclidian_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# if __name__ == "__main__":
#     graph = OrienteeringGraph(0)
#     with open("tsiligirides_problem_1_budget_05.txt") as node_file:
#         for index, line in enumerate(node_file):
#             if index == 0:
#                 graph.budget = line.split()[0]
#             else:
#                 x,y,value = line.split()
#                 x = float(x)
#                 y = float(y)
#                 value = int(value)
#                 graph.add_node(index, x, y, value)

#     # Add edges between each node
#     for node_a in graph.get_nodes():
#         for node_b in graph.get_nodes():
#             if node_a != node_b:  # No self-loops
#                 x1, y1 = graph.graph[node_a].x, graph.graph[node_a].y,
#                 x2, y2 = graph.graph[node_b].x, graph.graph[node_b].y,
#                 distance = calculate_euclidian_distance(x1, y1, x2, y2)
#                 graph.add_edge(node_a, node_b, distance)

#     graph.print_graph()