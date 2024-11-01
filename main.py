from orienteering_problem import Node, OrienteeringGraph, calculate_euclidian_distance
from mcts import mcts_run
from mcts_parallel_leaf import mcts_run_parallel_leaf
from mcts_parallel_root import mcts_run_parallel_root
from mcts_parallel_tree_global_mutex import mcts_run_parallel_tree
from mcts_parallel_tree_local_mutexes import mcts_run_parallel_tree_local_mutex
from mcts_parallel_tree_local_mutexes_spinlock import mcts_run_parallel_tree_local_mutex_spinlock
from mcts_parallel_tree_virtual_loss import mcts_run_parallel_tree_virtual_loss
import time

TSILIGIRIRIDES_1_MAXIMUM_REWARD = 285
SET_66_MAXIMUM_REWARD = 1660

def main():
    # Initialise orienteering graph and add nodes to the graph
    orienteering_graph = OrienteeringGraph(budget=0)
    file = "tsiligirides_1/tsiligirides_problem_1_budget_20.txt"
    if "tsiligirides" in file:
        maximum_reward = TSILIGIRIRIDES_1_MAXIMUM_REWARD
    elif "set_66" in file:
        maximum_reward = SET_66_MAXIMUM_REWARD

    with open(file) as node_file:
        for index, line in enumerate(node_file):
            if index == 0:
                orienteering_graph.budget = float(line.split()[0])
                
                # Set the maximum reward based on budget 
                if orienteering_graph.budget < 40:
                    maximum_reward /= 1.5
            else:
                x,y,value = line.split()
                x = float(x)
                y = float(y) 
                value = int(value) / maximum_reward
                orienteering_graph.add_node(index - 1, x, y, value)

    # Current Policy: Add edges between each node (Can be changed to different edge policy later)
    for node_a in orienteering_graph.get_nodes():
        for node_b in orienteering_graph.get_nodes():
            if node_a != node_b: 
                x1, y1 = orienteering_graph.graph[node_a].x, orienteering_graph.graph[node_a].y
                x2, y2 = orienteering_graph.graph[node_b].x, orienteering_graph.graph[node_b].y
                distance = calculate_euclidian_distance(x1, y1, x2, y2)
                if distance < 8:
                    orienteering_graph.add_edge(node_a, node_b, distance)

    # orienteering_graph.print_graph()
    # print("Average distance between nodes:", orienteering_graph.calculate_average_distance())
    print(f"Budget for this search is {orienteering_graph.budget}")

    #Start at first node with empty path and no time spent
    start_time = time.time()
    # best_mcts_node = mcts_run(graph=orienteering_graph, start_node_index=0)
    # best_mcts_node = mcts_run_parallel_leaf(graph=orienteering_graph, start_node_index=0)
    best_mcts_node = mcts_run_parallel_root(graph=orienteering_graph, start_node_index=0)
    # best_mcts_node = mcts_run_parallel_tree(graph=orienteering_graph, start_node_index=0)
    # best_mcts_node = mcts_run_parallel_tree_local_mutex(graph=orienteering_graph, start_node_index=0)
    # best_mcts_node = mcts_run_parallel_tree_virtual_loss(graph=orienteering_graph, start_node_index=0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("MCTS Search Time:", elapsed_time)
    print(f"MCTS Value achieved from MCTS search: {best_mcts_node.value}")
    print(f"Number of visits for node: {best_mcts_node.visits}")
    print(f"Path achieved from MCTS search: {best_mcts_node.path}")
    value_achieved = 0
    for node in best_mcts_node.path:
        value_achieved += orienteering_graph.get_node(node).value
    value_achieved *= maximum_reward
    print(f"OP Value achieved from MCTS search: {value_achieved}")

if __name__ == "__main__":
    main()