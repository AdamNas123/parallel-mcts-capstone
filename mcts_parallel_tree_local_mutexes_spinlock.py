import math
import random 

from orienteering_problem import OrienteeringGraph
from tree_node_spinlock import MCTSNode
from plot import setup_plot, update_plot, plot_final_path
# from multiprocessing import Lock
import concurrent.futures

# tree_lock = Lock()

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


# UCB Formula for Selection Phase - Select Child Nodes that maximise Upper Confidence Bound 1: 
def ucb1(node, exploration_constant):
    if node.visits == 0:
        return float('inf')
    # print("Node:", node.path, "Exploitation value:", node.value / node.visits, "Exploration value:", exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    ucb = (node.value / node.visits) + (exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    return ucb


#Selection and Expansion phase - Add children with global mutex
def select_and_expand(mcts_node: MCTSNode, graph: OrienteeringGraph, exploration_constant: float):
    while True:
        # Acquire the spinlock for the node
        print(f"Trying to acquire lock on node {mcts_node.op_node_index} for expansion")
        mcts_node.lock.acquire()
        print(f"Acquired lock on node {mcts_node.op_node_index} for expansion")
        try:
            if not mcts_node.children and mcts_node.possible_children:
                # Expand a new child
                next_child = mcts_node.possible_children.pop(0)
                new_child_node = MCTSNode(op_node_index=next_child, graph=graph, parent=mcts_node, path=mcts_node.path + [next_child])
                mcts_node.add_child(new_child_node)
                return new_child_node  # Return after expanding
            elif mcts_node.children:
                # Select the best child based on UCB1
                best_child = max(mcts_node.children, key=lambda node: ucb1(node, exploration_constant))
                mcts_node = best_child  # Move to the child node
            else:
                return mcts_node  # If no children, return the current node
        finally:
            # Always release the lock after acquiring it
            print(f"Releasing lock on node {mcts_node.op_node_index} after expansion")
            mcts_node.lock.release()

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
        print(f"Trying to acquire lock on node {current_node.op_node_index} for expansion")
        current_node.lock.acquire()
        print(f"Acquired lock on node {current_node.op_node_index} for expansion")
        
        try:
            # print("Backpropagating value of", reward, "to node with path:", current_node.path)
            current_node.update_value(reward)
        finally:
            print(f"Releasing lock on node {current_node.op_node_index} after expansion")
            current_node.lock.release()
        current_node = current_node.parent


def collect_visited_leaf_nodes(node):
    # Initialize list of leaf nodes
    leaf_nodes = []

    # Check if the current node is a leaf node
    if not node.children:
        leaf_nodes.append(node)
    else:
        # Recursively collect leaf nodes from children
        for child in node.children:
            leaf_nodes.extend(collect_visited_leaf_nodes(child))
    return leaf_nodes


def run_single_mcts(graph: OrienteeringGraph, root: MCTSNode, num_simulations: int, exploration_constant = 0.4):
    for simulation in range(num_simulations):
        mcts_node = root
        # Expansion
        mcts_node = select_and_expand(mcts_node,graph, exploration_constant)
        # Simulate from the current node
        reward = simulate_epsilon(graph, mcts_node)

        # Backpropagate the result
        backpropagate(mcts_node, reward)
        
        # Add possible children after simulation
        if not mcts_node.possible_children:
            add_possible_children(mcts_node, graph)
    print("Finished a single mcts")


# Calls all above functions to run the MCTS Search
def mcts_run_parallel_tree_local_mutex_spinlock(graph: OrienteeringGraph, start_node_index=0, num_simulations=50000, num_threads=4):
    fig, ax, G, pos = setup_plot(graph)
    
    # Selection for first node (root node)
    root = MCTSNode(op_node_index=start_node_index, graph=graph, is_root=True, lock=True)
    add_possible_children(root, graph)
    exploration_constant = 0.4 #Exploration constant
    print("Exploration constant:", exploration_constant)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_single_mcts, graph, root, num_simulations) for _ in range(num_threads)]
        # Ensure that all futures complete
        concurrent.futures.wait(futures) 
        for future in futures:
            future.result()  # To raise any exceptions encountered during execution

    # Collect all leaf nodes
    leaf_nodes = collect_visited_leaf_nodes(root)

    # Return best leaf node based on value first, then visits
    # best_node = max(leaf_nodes, key=lambda n: (n.value, n.visits))
    best_node = max((n for n in leaf_nodes if n.visits > 0), key=lambda n: (n.value), default=None)
    plot_final_path(ax, G, pos, graph, best_node.path, filename="final_paths/final_path_parallel_tree_local_mutex_spinlock_budget_40.png")
    return best_node