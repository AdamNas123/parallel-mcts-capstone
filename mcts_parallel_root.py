import math
import random 

from orienteering_problem import OrienteeringGraph
from tree_node import MCTSNode
from plot import setup_plot, update_plot, plot_final_path, plot_rewards_parallel

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

# Add Possible Neighbours (Unexpanded Children) - Used to Select best child node and expand into tree
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


# Simulation Phase - Randomly simulate path from current node until budget limit is reached or no more nodes available
def simulate(graph: OrienteeringGraph, mcts_node: MCTSNode):
    current_index = mcts_node.op_node_index
    total_reward = sum(graph.get_node(node_index).value for node_index in mcts_node.path)
    remaining_budget = graph.budget - mcts_node.get_path_distance(graph)
    visited = set(mcts_node.path)
    path = mcts_node.path[:]

    while remaining_budget > 0:
        neighbours = {k: v for k, v in graph.get_neighbours(current_index).items() if k not in visited}
        if not neighbours:
            break

        #GREEDILY choose next neighbour based on value of neighbour node / distance
        # next_node = max(neighbours.keys(), key=lambda n: graph.get_node(n).value / neighbours[n]) #Divide by distance
        # distance = neighbours[next_node]

        # OR uncomment below lines to RANDOMLY choose next neighbour from unvisited neighbours 
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
    print(f"Simulated Path: {path}, Reward: {total_reward}, Remaining Budget: {remaining_budget}")
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

# After merging root children, collect leaf nodes from each tree and select the best
def parallel_mcts_select_best_node(aggregated_root, graph):
    # Initialize a list to hold all leaf nodes across all trees
    all_leaf_nodes = collect_visited_leaf_nodes(aggregated_root)

    # Check if we have any leaf nodes with visits
    if not all_leaf_nodes:
        raise ValueError("No leaf nodes were visited across the parallel trees.")

    # Select the best leaf node based on value (and optionally visits)
    best_leaf_node = max(
        (n for n in all_leaf_nodes if n.visits > 0),  # Only consider nodes with visits
        key=lambda n: n.value,  # Compare nodes based on value (or customize if needed)
        default=None
    )

    return best_leaf_node

# Calls all above functions to run the MCTS Search
# Function to run a single MCTS instance
def run_single_mcts(graph: OrienteeringGraph, root: MCTSNode, num_simulations: int, exploration_constant=0.4):
    intermediate_rewards = []

    for simulation in range(num_simulations):
        # print("On iteration", simulation)
        mcts_node = root
        if not mcts_node.children and mcts_node.possible_children:
            next_child = mcts_node.possible_children.pop(0)
            new_child_node = MCTSNode(op_node_index=next_child, graph=graph, parent=mcts_node, path=mcts_node.path + [next_child])
            mcts_node.add_child(new_child_node)
            mcts_node = new_child_node
        else:
            while True:
                if mcts_node.possible_children:
                    next_child = mcts_node.possible_children.pop(0)
                    new_child_node = MCTSNode(op_node_index=next_child, graph=graph, parent=mcts_node, path=mcts_node.path + [next_child])
                    mcts_node.add_child(new_child_node)
                    mcts_node = new_child_node
                    break
                elif mcts_node.children:
                    mcts_node = max(mcts_node.children, key=lambda node: ucb1(node, exploration_constant))
                else:
                    break

        reward = simulate_epsilon(graph, mcts_node)
        backpropagate(mcts_node, reward)
        add_possible_children(mcts_node, graph)
        intermediate_rewards.append(reward)
        # Yield intermediate rewards every `yield_interval` iterations
        # if (simulation + 1) % yield_interval == 0:
        #     intermediate_rewards.append(reward)
        #     print(f"Simulation {simulation + 1}: Intermediate reward added - {reward}")
        #     yield intermediate_rewards

    
    print("Finished a single mcts")
    yield intermediate_rewards, root

# Aggregates statistics from multiple root nodes
def aggregate_child_nodes(aggregated_node, child_node, path_to_node_map, graph):
    # Look for a corresponding child node in the map for faster lookup
    corresponding_child = path_to_node_map.get(tuple(child_node.path), None)

    if corresponding_child:
        # print("Found Corresponding child with path", corresponding_child.path)
        corresponding_child.visits += child_node.visits
        corresponding_child.value += child_node.value

        # Recursively aggregate children of the current node
        for grandchild in child_node.children:
            aggregate_child_nodes(corresponding_child, grandchild, path_to_node_map, graph)
    else:
        # Instead of deep copying the entire child_node, only add essential values and children
        # Create a new shallow copy of the node without its children
        # print("No corresponding child found, creating new child with path", child_node.path)
        new_child = MCTSNode(
            op_node_index=child_node.op_node_index,
            parent=aggregated_node,
            graph=graph,
            path=child_node.path[:],  # Shallow copy of the path
        )
        new_child.visits = child_node.visits
        new_child.value=child_node.value     # Copy values
        aggregated_node.add_child(new_child)
        path_to_node_map[tuple(new_child.path)] = new_child  # Update the map with new child

        # Continue aggregation with the children of the newly added node
        for grandchild in child_node.children:
            aggregate_child_nodes(new_child, grandchild, path_to_node_map, graph)

def aggregate_root_results(root_nodes, graph):
    aggregated_root = deepcopy(root_nodes[0])  # Create a deep copy of the first root (shallow copy below)
    path_to_node_map = {tuple(child.path): child for child in aggregated_root.children}  # Map paths to children

    # Iterate over all other root nodes
    for i in range(1, len(root_nodes)):
        for child in root_nodes[i].children:
            # Aggregate the entire subtree of each child recursively
            aggregate_child_nodes(aggregated_root, child, path_to_node_map, graph)

    return aggregated_root




# Main MCTS function with root parallelization
def mcts_run_parallel_root(graph: OrienteeringGraph, start_node_index=0, num_simulations=205000, num_threads=4):
    fig, ax, G, pos = setup_plot(graph)
    exploration_constant=0.4
    all_rewards_over_time = []
    thread_rewards = [[] for _ in range(num_threads)] 
    final_root_nodes = [] 

    # Create independent root nodes for each thread
    root_nodes = [MCTSNode(op_node_index=start_node_index, graph=graph, is_root=True) for _ in range(num_threads)]
    for root in root_nodes:
        add_possible_children(root, graph)

    # Parallel execution of multiple MCTS instances
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_single_mcts, graph, root, num_simulations, exploration_constant) for root in root_nodes]
        # parallel_roots = [f.result() for f in futures]

    # Collect intermediate results over time and final root nodes
        for thread_index, future in enumerate(as_completed(futures)):
            for result in future.result():
                intermediate_rewards, final_root = result
                final_root_nodes.append(final_root)

                thread_rewards[thread_index].extend(intermediate_rewards)
                
                for i, reward in enumerate(intermediate_rewards):
                    if i >= len(all_rewards_over_time):
                        all_rewards_over_time.append([])  # Initialize a new list for each new iteration index
                    all_rewards_over_time[i].append(reward) 

    # Aggregate the results of all parallel root nodes
    aggregated_root = aggregate_root_results(final_root_nodes, graph)
    
    # After all simulations, select the best node from all root nodes based on leaf nodes
    best_node = parallel_mcts_select_best_node(aggregated_root, graph)
    
    plot_final_path(ax, G, pos, graph, best_node.path, filename="final_path_parallel_root.png")
    averaged_rewards = [sum(rewards) / len(rewards) for rewards in all_rewards_over_time]
    plot_rewards_parallel(thread_rewards, averaged_rewards, filename=f"logs/parallel_root/results/simulations_{num_simulations // 1000}k.png")
    return best_node