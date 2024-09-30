import math
import random 

from orienteering_problem import Node, OrienteeringGraph
from tree_node import MCTSNode


# Selection Phase - Select Child Nodes that maximise Upper Confidence Bound 1: 
def ucb1(node, exploration_constant):
    if node.visits == 0:
        return float('inf')
    # print("Exploitation value:", node.value / node.visits, "Exploration value:", exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    ucb = (node.value / node.visits) + (exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    return ucb


# Expansion Phase - If selected node is not terminal, expand it by adding child node for each valid action
def expand_tree(mcts_node: MCTSNode, graph: OrienteeringGraph):
    current_index = mcts_node.op_node_index
    current_path_distance = mcts_node.get_path_distance(graph)
    neighbours = graph.get_neighbours(current_index)
    new_nodes = []

    # Check if it's the first expansion (root node)
    if mcts_node.is_root:
        # Add all valid neighbours within budget as children
        for neighbour_index, distance in neighbours.items():
            if (current_path_distance + distance) <= graph.budget and neighbour_index not in mcts_node.path:
                new_path = mcts_node.path + [neighbour_index]
                child_node = MCTSNode(op_node_index=neighbour_index, graph=graph,parent=mcts_node, path=new_path)
                mcts_node.add_child(child_node)
                new_nodes.append(neighbour_index)
        print(f"Added all possible child nodes {new_nodes} to the root node {mcts_node.path}")

    else:
        # Sort neighbours by their value, and only add the top 3 or fewer within the budget
        valid_neighbours = [(neighbour_index, distance) for neighbour_index, distance in neighbours.items() 
                            if (current_path_distance + distance) <= graph.budget and neighbour_index not in mcts_node.path]
        
        # Sort by the node values in descending order
        sorted_neighbours = sorted(valid_neighbours, key=lambda x: graph.get_node(x[0]).value, reverse=True)

        # Add up to 3 neighbours
        count = 0
        for neighbour_index, distance in sorted_neighbours:
            if count < 3:
                new_path = mcts_node.path + [neighbour_index]
                child_node = MCTSNode(op_node_index=neighbour_index, graph=graph,parent=mcts_node, path=new_path)
                mcts_node.add_child(child_node)
                new_nodes.append(neighbour_index)
                count += 1
            else:
                break
        
        print(f"Added child nodes {new_nodes} to the parent {mcts_node.path}")


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
        next_node = max(neighbours.keys(), key=lambda n: graph.get_node(n).value / neighbours[n]) #Divide by distance
        distance = neighbours[next_node]

        #Check if distance to node is within budget
        if distance > remaining_budget:
            break
        current_index = next_node
        remaining_budget -= distance
        total_reward += graph.get_node(current_index).value
        visited.add(current_index)
        path.append(current_index)
    print(f"Simulated Path: {path}, Reward: {total_reward}, Remaining Budget: {remaining_budget}")
    return total_reward


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
    print("Leaf Nodes:", leaf_nodes)
    return leaf_nodes

# Calls all above functions to run the MCTS Search
def mcts_run(graph: OrienteeringGraph, start_node_index=0, num_simulations=100000):
    # Selection for first node (root node)
    root = MCTSNode(op_node_index=start_node_index, graph=graph, is_root=True)
    exploration_constant = 1.414 # Starting exploration constant

    for _ in range(num_simulations):
        #Selection for following nodes by calculating UCB
        mcts_node = root
        while mcts_node.children:
            #Select the child node that maximises ucb1
            mcts_node = max(mcts_node.children, key=lambda node: ucb1(node, exploration_constant))
            print("Selected node ", mcts_node.path, "with value:", mcts_node.value)

        #Expansion
        expand_tree(mcts_node, graph)

        #Simulation
        reward = simulate(graph, mcts_node)

        #Backpropagations
        backpropagate(mcts_node,reward)

    # Collect all leaf nodes
    leaf_nodes = collect_visited_leaf_nodes(root)
    
    # If there are no leaf nodes, fallback to children of root (unlikely)
    # if not leaf_nodes:
    #     leaf_nodes = root.children

    # Return best leaf node based on value first, then visits
    # best_node = max(leaf_nodes, key=lambda n: (n.value, n.visits))
    best_node = max((n for n in leaf_nodes if n.visits > 0), key=lambda n: (n.value, n.visits), default=None)
    
    return best_node
