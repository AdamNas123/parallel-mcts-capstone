import math
import random 

from orienteering_problem import Node, OrienteeringGraph
from tree_node import MCTSNode


# Selection Phase - Select Child Nodes that maximise Upper Confidence Bound 1: 
def ucb1(node, exploration_constant=1.414):
    if node.visits == 0:
        return float('inf')
    return (node.value / node.visits) + exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits)


# Expansion Phase - If selected node is not terminal, expand it by adding child node for each valid action
def expand_tree(mcts_node: MCTSNode, graph: OrienteeringGraph, visited):
    current_index = mcts_node.node_index
    current_path_distance = mcts_node.get_path_distance(graph)

    for neighbour_index, distance in graph.get_neighbours(current_index).items():
        if neighbour_index not in visited and distance <= (graph.budget - current_path_distance):
            new_path = mcts_node.path + [neighbour_index]
            child_node = MCTSNode(neighbour_index, parent=mcts_node, path=new_path)
            mcts_node.add_child(child_node)

# Simulation Phase - Randomly simulate path from current node until budget limit is reached or no more nodes available
def simulate(graph: OrienteeringGraph, start_node_index: int):
    current_index = start_node_index
    total_reward = graph.get_node(current_index).value
    remaining_budget = graph.budget
    visited = {current_index}
    path = [current_index]

    while remaining_budget > 0:
        neighbours = graph.get_neighbours(current_index)
        if not neighbours:
            break

        #Randomly choose neighbour to visit
        next_node = random.choice(list(neighbours.keys()))
        distance = neighbours[next_node]

        #Check if distance to node is within budget
        if distance > remaining_budget:
            break
        current_index = next_node
        remaining_budget -= distance
        total_reward += graph.get_node(current_index).value
        visited.add(current_index)
        path.append(current_index)
    
    # print(f"Simulated Path: {path}, Reward: {total_reward}, Remaining Budget: {remaining_budget}")
    return total_reward, path


# Backpropagation Phase - After Simulation, propagate score back up tree. Updates total score and visit count for each node.
def backpropagate(mcts_node: MCTSNode, reward):
    current_node = mcts_node
    while current_node is not None:
        current_node.update_value(reward)
        current_node = current_node.parent


# Calls all above functions to run the MCTS Search
def mcts_run(graph: OrienteeringGraph, start_node_index=0, num_simulations=1000):
    root = MCTSNode(start_node_index)

    for _ in range(num_simulations):
        #Selection by calculating UCB
        mcts_node = root
        visited = {mcts_node.node_index}
        while mcts_node.children:
            #Select the child node that maximises ucb1
            mcts_node = max(mcts_node.children, key=ucb1)
            visited.add(mcts_node.node_index)

        #Expansion
        expand_tree(mcts_node, graph, visited)

        #Simulation
        reward, simulated_path = simulate(graph, mcts_node.node_index)

        #Backpropagation
        backpropagate(mcts_node,reward)

    #Return best path based on visit counts or values
    best_node = max(root.children, key = lambda n: n.visits)
    return best_node
