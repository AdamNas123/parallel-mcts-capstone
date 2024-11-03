import math
import random 
import time

from orienteering_problem import OrienteeringGraph
from tree_node import MCTSNode
from plot import setup_plot, plot_final_path, plot_rewards_parallel, plot_rewards_average, plot_rewards_time_parallel
from multiprocessing import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

tree_lock = Lock()
rewards_queue = Queue()
time_queue = Queue()

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


# UCB Formula for Selection Phase - Select Child Nodes that maximise Upper Confidence Bound 1: 
def ucb1(node, exploration_constant):
    if node.visits == 0:
        return float('inf')
    ucb = (node.value / node.visits) + (exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits))
    return ucb


#Selection and Expansion phase - Add children with global mutex
def select_and_expand(mcts_node: MCTSNode, graph: OrienteeringGraph, exploration_constant: float):
    with tree_lock:
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
    return mcts_node

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
            next_node = max(neighbours.keys(), key=lambda n: graph.get_node(n).value / neighbours[n])

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
    with tree_lock:
        while current_node is not None:
            current_node.update_value(reward)
            current_node = current_node.parent


def collect_visited_leaf_nodes(node):
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
    # intermediate_rewards = []

    for _ in range(num_simulations):
        # Selection and Expansion
        mcts_node = select_and_expand(root,graph, exploration_constant)
        
        # Simulate from the current node
        reward = simulate_epsilon(graph, mcts_node)
        rewards_queue.put(reward)
        timestamp = time.time()
        time_queue.put(timestamp)
        # intermediate_rewards.append(reward)

        # Backpropagate the result
        backpropagate(mcts_node, reward)
        
        # Add possible children after simulation
        if not mcts_node.possible_children:
            add_possible_children(mcts_node, graph)

    print("Finished a single mcts")
    # yield intermediate_rewards


# Calls all above functions to run the MCTS Search
def mcts_run_parallel_tree(graph: OrienteeringGraph, start_node_index=0, num_simulations=31250, num_threads=16):
    # _, ax, G, pos = setup_plot(graph)
    ordered_rewards = []
    time_log = []
    exploration_constant = 0.4
    # all_rewards_over_time = []
    # thread_rewards = [[] for _ in range(num_threads)] 

    start_time = time.time()
    # Selection for first node (root node)
    root = MCTSNode(op_node_index=start_node_index, graph=graph, is_root=True)
    add_possible_children(root, graph)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_single_mcts, graph, root, num_simulations, exploration_constant) for _ in range(num_threads)]

        for future in as_completed(futures):
            future.result()

        # for thread_index, future in enumerate(as_completed(futures)):
        #     for intermediate_results in future.result():
        #         thread_rewards[thread_index].extend(intermediate_results)

        #         for i, reward in enumerate(intermediate_results):
        #             if i >= len(all_rewards_over_time):
        #                 all_rewards_over_time.append([]) 
        #             all_rewards_over_time[i].append(reward) 

    
    while not rewards_queue.empty():
        ordered_rewards.append(rewards_queue.get())
        timestamp = time_queue.get() - start_time
        time_log.append(timestamp)
    
    # Collect all leaf nodes
    leaf_nodes = collect_visited_leaf_nodes(root)

    # Return best leaf node based on value first, then visits
    # best_node = max(leaf_nodes, key=lambda n: (n.value, n.visits))
    best_node = max((n for n in leaf_nodes if n.visits > 0), key=lambda n: (n.value), default=None)

    #Plot final chosen path
    # plot_final_path(ax, G, pos, graph, best_node.path, filename="final_paths/final_path_parallel_tree_budget_40.png")
    
    #Plot rollout rewards
    plot_rewards_parallel(ordered_rewards, filename=f"logs/parallel_tree_global_mutex/threads/results/threads_{num_threads}_budget_{graph.budget}_simulations_{num_simulations*num_threads}.png", step=12500)
    # averaged_rewards = [sum(rewards) / len(rewards) for rewards in all_rewards_over_time]
    # plot_rewards_average(thread_rewards, averaged_rewards, filename=f"logs/parallel_tree_global_mutex/results/budget_{graph.budget}_simulations_{num_simulations*num_threads}.png", step=5000)
    
    #Plot Time Rewards
    plot_rewards_time_parallel(time_log, ordered_rewards, filename=f"logs/parallel_tree_global_mutex/threads/results_time/threads_{num_threads}_budget_{graph.budget}_simulations_{num_simulations*num_threads}.png", step=12500)

    return best_node