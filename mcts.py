import math
import random 

from orienteering_problem import OrienteeringGraph
from tree_node import MCTSNode
from plot import setup_plot, update_plot, plot_final_path, plot_rewards



# Add Possible Neighbours (Unexpanded Children) - Used to Select best child node and expand into tree
"""def add_possible_children(mcts_node: MCTSNode, graph: OrienteeringGraph):
    current_index = mcts_node.op_node_index
    current_path_distance = mcts_node.get_path_distance(graph)
    neighbours = graph.get_neighbours(current_index)
    new_nodes = []

    # Check if it's the root node
    # if mcts_node.is_root:
    #     # Add all valid neighbours within budget as children
    #     for neighbour_index, distance in neighbours.items():
    #         if (current_path_distance + distance) <= graph.budget and neighbour_index not in mcts_node.path:
    #             mcts_node.add_possible_child(neighbour_index)
    #             new_nodes.append(neighbour_index)
    #     # print(f"Added all possible child nodes {new_nodes} to the root node {mcts_node.path}")

    # else:
    # Sort neighbours by their value, and only add the top 3 or fewer within the budget
    valid_neighbours = [(neighbour_index, distance) for neighbour_index, distance in neighbours.items() 
                        if (current_path_distance + distance) <= graph.budget and neighbour_index not in mcts_node.path]
    
    # Sort by the node value-distance ratios in descending order
    sorted_neighbours = sorted(valid_neighbours, key=lambda x: graph.get_node(x[0]).value / x[1], reverse=True)

    # Add up to 3 neighbours
    count = 0
    for neighbour_index, distance in sorted_neighbours:
        if count < 2:
            mcts_node.add_possible_child(neighbour_index)
            new_nodes.append(neighbour_index)
            count += 1
        else:
            break
    
    print(f"Added possible child nodes {new_nodes} to the parent {mcts_node.path}")
"""

def add_possible_children(mcts_node: MCTSNode, graph: OrienteeringGraph):
    current_index = mcts_node.op_node_index
    current_path_distance = mcts_node.get_path_distance(graph)
    neighbours = graph.get_neighbours(current_index)
    new_nodes = []

    # Add all valid neighbours within budget as possible children
    for neighbour_index, distance in neighbours.items():
        if (current_path_distance + distance) <= graph.budget and neighbour_index not in mcts_node.path:
            mcts_node.add_possible_child(neighbour_index)
            new_nodes.append(neighbour_index)
    # print(f"Added all possible child nodes {new_nodes} to the node {mcts_node.path}")



# Selection Phase - Select Child Nodes that maximise Upper Confidence Bound 1: 
def ucb1(node, exploration_constant):
    if node.visits == 0:
        return float('inf')
    # print("Node:", node.path, "Exploitation value:", node.value / node.visits, "Exploration value:", exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    ucb = (node.value / node.visits) + (exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    return ucb


# Expansion Phase - If selected node is not terminal, expand it by adding child node for each valid action
# def expand_tree(mcts_node: MCTSNode, graph: OrienteeringGraph):
    


# Simulation Phase - Randomly simulate path from current node until budget limit is reached or no more nodes available
def simulate(graph: OrienteeringGraph, mcts_node: MCTSNode):
    current_index = mcts_node.op_node_index
    total_reward = sum(graph.get_node(node_index).value for node_index in mcts_node.path)
    remaining_budget = graph.budget - mcts_node.get_path_distance(graph)
    visited = set(mcts_node.path)
    path = mcts_node.path[:]

    while remaining_budget > 0:
        # Should this neighbours be the MCTS nodes child instead of graph neighbours
        neighbours = {k: v for k, v in graph.get_neighbours(current_index).items() if k not in visited}
        if not neighbours:
            break

        #GREEDILY choose next neighbour based on value of neighbour node / distance
        # next_node = max(neighbours.keys(), key=lambda n: graph.get_node(n).value / neighbours[n]) #Divide by distance
        # distance = neighbours[next_node]

        # OR randomly choose next neighbour from unvisited neighbours 
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
    return total_reward


# Simulate with Epsilon-Greedy Approach
def simulate_epsilon(graph: OrienteeringGraph, mcts_node: MCTSNode, epsilon=0.3):
    current_index = mcts_node.op_node_index
    total_reward = sum(graph.get_node(node_index).value for node_index in mcts_node.path)
    remaining_budget = graph.budget - mcts_node.get_path_distance(graph)
    visited = set(mcts_node.path)
    path = mcts_node.path[:]

    while remaining_budget > 0:
        neighbours = {k: v for k, v in graph.get_neighbours(current_index).items() if k not in visited}
        if not neighbours:
            break

        # Use epsilon-greedy strategy to choose the next node
        if random.random() < epsilon:
            next_node = random.choice(list(neighbours.keys()))  # Randomly explore
        else:
            next_node = max(neighbours.keys(), key=lambda n: graph.get_node(n).value / neighbours[n])  # Greedy

        distance = neighbours[next_node]
        if distance > remaining_budget:
            break
        current_index = next_node
        remaining_budget -= distance
        total_reward += graph.get_node(current_index).value
        visited.add(current_index)
        path.append(current_index)

    # print(f"Simulated Path: {path}, Reward: {total_reward}, Remaining Budget: {remaining_budget}")
    return total_reward


# Backpropagation Phase - After Simulation, propagate score back up tree. Updates total score and visit count for each node.
def backpropagate(mcts_node: MCTSNode, reward):
    current_node = mcts_node
    while current_node is not None:
        current_node.update_value(reward)
        # print("Back propagating value of", reward, "to", current_node.path, ". Creating total value:", current_node.value, ". Visits are now:", current_node.visits)
        current_node = current_node.parent


def collect_visited_leaf_nodes(node):
    # Initialize list of leaf nodes
    leaf_nodes = []

    # Check if the current node is a leaf node
    # print("Node Index:", node.op_node_index, ", Parent:", node.parent, ", Visits:", node.visits, ", Value:", node.value, ", Children:", node.children, ", Path:", node.path) 
    if not node.children:
        # print("A Leaf Node!")
        leaf_nodes.append(node)
    else:
        # Recursively collect leaf nodes from children
        for child in node.children:
            leaf_nodes.extend(collect_visited_leaf_nodes(child))
    # print("Leaf Nodes:", leaf_nodes)
    return leaf_nodes

# Calls all above functions to run the MCTS Search
def mcts_run(graph: OrienteeringGraph, start_node_index=0, num_simulations=1500000):
    fig, ax, G, pos = setup_plot(graph)
    
    # Selection for first node (root node)
    root = MCTSNode(op_node_index=start_node_index, graph=graph, is_root=True)
    add_possible_children(root, graph)
    exploration_constant = 0.4
    print("Exploration constant:", exploration_constant)
    rewards_log = []
    for _ in range(num_simulations):
        #Selection and expansion for following nodes by calculating UCB
        mcts_node = root
        
        if not mcts_node.children and mcts_node.possible_children:
            next_child = mcts_node.possible_children.pop(0)
            new_child_node = MCTSNode(op_node_index=next_child, graph=graph, parent=mcts_node, path=mcts_node.path + [next_child])
            mcts_node.add_child(new_child_node)
            mcts_node = new_child_node  # Move to the newly added child 
        else:
            # If the node has any possible children, select the first possible child.
            while True:
                # If there are possible children, expand them
                if mcts_node.possible_children:
                    next_child = mcts_node.possible_children.pop(0)
                    new_child_node = MCTSNode(op_node_index=next_child, graph=graph, parent=mcts_node, path=mcts_node.path + [next_child])
                    mcts_node.add_child(new_child_node)
                    mcts_node = new_child_node  # Move to the newly added child
                    break  # Expansion finished, move to simulation
                # Else, select the child with the best UCB value
                elif mcts_node.children:
                    mcts_node = max(mcts_node.children, key=lambda node: ucb1(node, exploration_constant))
                    # print("Selected visited node", mcts_node.path, "with value:", mcts_node.value)
                else:
                    # No children or possible children left, break out to simulate
                    break

        #Simulation
        reward = simulate_epsilon(graph, mcts_node)
        rewards_log.append(reward)
        #Backpropagations
        backpropagate(mcts_node,reward)

        #Before going back to root, add possible children of child node
        add_possible_children(mcts_node=mcts_node, graph=graph)

    # Collect all leaf nodes
    leaf_nodes = collect_visited_leaf_nodes(root)
    
    # If there are no leaf nodes, fallback to children of root (unlikely)
    # if not leaf_nodes:
    #     leaf_nodes = root.children

    # Return best leaf node based on value first, then visits
    # best_node = max(leaf_nodes, key=lambda n: (n.value, n.visits))
    best_node = max((n for n in leaf_nodes if n.visits > 0), key=lambda n: (n.value), default=None)
    plot_final_path(ax, G, pos, graph, best_node.path, filename=f"final_path_budget_{graph.budget}.png")
    plot_rewards(rewards_log, filename=f"logs/base_mcts/rewards/budget_{graph.budget}_simulations_{num_simulations}.png", step=15000)
    return best_node
