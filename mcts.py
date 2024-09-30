import math
import random 

from orienteering_problem import Node, OrienteeringGraph
from tree_node import MCTSNode


# Selection Phase - Select Child Nodes that maximise Upper Confidence Bound 1: 
def ucb1(node, exploration_constant):
    if node.visits == 0:
        # print("UCB for node", node.path, ": infinite")
        return float('inf')
    # print("Current node value:", node.value)
    # print("Exploitation value:", node.value / node.visits)
    # print("Exploration value:", exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    ucb = (node.value / node.visits) + (exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    # print("UCB for node", node.path, ":", ucb)
    return ucb


# Expansion Phase - If selected node is not terminal, expand it by adding child node for each valid action
# def expand_tree(mcts_node: MCTSNode, graph: OrienteeringGraph, visited):
#     current_index = mcts_node.op_node_index
#     current_path_distance = mcts_node.get_path_distance(graph)
#     # print("Current path distance: ", current_path_distance)
#     for neighbour_index, distance in graph.get_neighbours(current_index).items():
#         if neighbour_index not in visited and (current_path_distance + distance) <= graph.budget:
#             new_path = mcts_node.path + [neighbour_index]
#             child_node = MCTSNode(neighbour_index, graph, parent=mcts_node, path=new_path)
#             mcts_node.add_child(child_node)
#             # print(f"Added child: {neighbour_index}, Path: {new_path}")
#             # print(f"This path has a distance of {child_node.get_path_distance(graph)}")

def expand_tree(mcts_node: MCTSNode, graph: OrienteeringGraph, first_simulated_node: int):
    current_index = mcts_node.op_node_index
    current_path_distance = mcts_node.get_path_distance(graph)
    new_nodes = []
    neighbours = {k: v for k, v in graph.get_neighbours(current_index).items() if k not in mcts_node.path}

    # print("Current path distance: ", current_path_distance)
    # for neighbour_index, distance in graph.get_neighbours(current_index).items():
    #     if (current_path_distance + distance) <= graph.budget and neighbour_index not in mcts_node.path:
    #         new_path = mcts_node.path + [neighbour_index]
    #         child_node = MCTSNode(neighbour_index, graph=graph, parent=mcts_node, path=new_path)
    #         mcts_node.add_child(child_node)
    #         new_nodes.append(neighbour_index)
    if first_simulated_node:
        new_path = mcts_node.path + [first_simulated_node]
        child_node = MCTSNode(first_simulated_node, graph=graph, parent=mcts_node, path=new_path)
        mcts_node.add_child(child_node)
        new_nodes.append(first_simulated_node)
        print(f"Added first simulated child nodes {new_nodes} to the parent {mcts_node.path}")
    elif neighbours: 
        random_neighbour = random.choice(list(neighbours.keys()))
        distance = neighbours[random_neighbour]
        if (current_path_distance + distance) <= graph.budget:
            new_path = mcts_node.path + [random_neighbour]
            child_node = MCTSNode(random_neighbour, graph=graph, parent=mcts_node, path=new_path)
            mcts_node.add_child(child_node)
            new_nodes.append(random_neighbour)
            print(f"Added random child nodes {new_nodes} to the parent {mcts_node.path}")
            # print(f"Added child: {neighbour_index}, Path: {new_path}")
            # print(f"This path has a distance of {child_node.get_path_distance(graph)}")
    

# Simulation Phase - Randomly simulate path from current node until budget limit is reached or no more nodes available
def simulate(graph: OrienteeringGraph, mcts_node: MCTSNode):
    current_index = mcts_node.op_node_index
    total_reward = 0
    remaining_budget = graph.budget - mcts_node.get_path_distance(graph)
    visited = set(mcts_node.path)
    path = mcts_node.path[:]

    while remaining_budget > 0:
        # Should this neighbours be the MCTS nodes child instead of graph neighbours
        neighbours = {k: v for k, v in graph.get_neighbours(current_index).items() if k not in visited}
        if not neighbours:
            break

        #Randomly choose neighbour to visit
        # next_node = random.choice(list(neighbours.keys()))
        next_node = max(neighbours.keys(), key=lambda n: graph.get_node(n).value / neighbours[n])
        distance = neighbours[next_node]

        #Check if distance to node is within budget
        if distance > remaining_budget:
            break
        current_index = next_node
        remaining_budget -= distance
        total_reward += graph.get_node(current_index).value
        visited.add(current_index)
        path.append(current_index)
    first_simulated_node = path[len(mcts_node.path)] if len(path) > 0 else None
    # first_simulated_node = path[0] if len(path) > 0 else None
    print(f"Simulated Path: {path}, Reward: {total_reward}, Remaining Budget: {remaining_budget}")
    return total_reward, first_simulated_node


# Backpropagation Phase - After Simulation, propagate score back up tree. Updates total score and visit count for each node.
def backpropagate(mcts_node: MCTSNode, reward):
    current_node = mcts_node
    while current_node is not None:
        current_node.update_value(reward)
        print("Back propagating value of", reward, "to", current_node.path, ". Creating total value:", current_node.value, ". Visits are now:", current_node.visits)
        current_node = current_node.parent


def collect_visited_leaf_nodes(node):
    # Initialize list of leaf nodes
    leaf_nodes = []

    # Check if the current node is a leaf node
    # print("Node Index:", node.op_node_index, ", Parent:", node.parent, ", Visits:", node.visits, ", Value:", node.value, ", Children:", node.children, ", Path:", node.path) 
    if not node.children:
        if node.visits > 0:
        # print("A Leaf Node!")
            leaf_nodes.append(node)
    else:
        # Recursively collect leaf nodes from children
        for child in node.children:
            leaf_nodes.extend(collect_visited_leaf_nodes(child))
    # print("Leaf Nodes:", leaf_nodes)
    return leaf_nodes

# def get_best_node(node):
#     while node.children:
#         best_score = 0
#         best_child = -1
#         for child in node.children:
#             score = child.value
#             if best_child == -1 or (score > best_score):
#                 best_child = child
#                 best_score = score
#         node = best_child
#     return node

# Calls all above functions to run the MCTS Search
def mcts_run(graph: OrienteeringGraph, start_node_index=0, num_simulations=100000):
    # Selection for first node (root node)
    root = MCTSNode(start_node_index, graph)
    exploration_constant = 1.414 # Starting exploration constant
    first_simulated_node = None
    # decay_factor = 1.0  # Adjust this to control how quickly the exploration constant decays

    for _ in range(num_simulations):
        #Selection for following nodes by calculating UCB
        mcts_node = root
        # exploration_constant = initial_exploration_constant / (1 + simulation_count / decay_factor)
        # visited = {mcts_node.op_node_index}
        while mcts_node.children:
            #Select the child node that maximises ucb1
            mcts_node = max(mcts_node.children, key=lambda node: ucb1(node, exploration_constant))
            print("Selected node ", mcts_node.path, "with value:", mcts_node.value)
            # visited.add(mcts_node.op_node_index)

        #Expansion
        # expand_tree(mcts_node, graph, visited)
        expand_tree(mcts_node, graph, first_simulated_node=first_simulated_node)

        #Simulation
        reward, first_simulated_node = simulate(graph, mcts_node)

        #Backpropagations
        # print("The reward being backpropagated is", reward)
        backpropagate(mcts_node,reward)
    
    # best_node = get_best_node(root)
      
    # #Return best path based on visit counts or values
    # best_node = max(root.children, key=lambda n: (n.value, n.visits))

    # Collect all leaf nodes
    # Should change to just find child with best score and continue through its children
    leaf_nodes = collect_visited_leaf_nodes(root)
    

    # If there are no leaf nodes, fallback to children of root (unlikely)
    # if not leaf_nodes:
    #     leaf_nodes = root.children

    # Return best leaf node based on value first, then visits
    # best_node = max(leaf_nodes, key=lambda n: (n.value, n.visits))
    # best_node = max(leaf_nodes, key=lambda n: n.value)
    best_node = max((n for n in leaf_nodes if n.visits > 0), key=lambda n: (n.value, n.visits), default=None)
    
    return best_node
