import matplotlib.pyplot as plt
from orienteering_problem import OrienteeringGraph
import networkx as nx
import numpy as np

# Define color mapping based on node value
def get_node_colour(value):
    if value == 0:
        return 'lightgrey'  # Greyed out for value 0
    elif value == 5:
        return 'darkgrey'  # Darker grey for value 5
    elif value == 10:
        return 'lightgreen'  # Light green for value 10
    elif value == 15:
        return 'green'  # Normal green for value 15
    else:
        return 'white'  # Default color if the value is not 0, 5, 10, or 15

# Plot setup
def setup_plot(graph: OrienteeringGraph):
    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots()
    G = nx.Graph()
    
    # Add nodes to the graph with coordinates
    for node in graph.get_nodes():
        G.add_node(node, pos=(graph.graph[node].x, graph.graph[node].y))
    
    # Add edges to the graph (using adjacency list)
    for node in graph.get_nodes():
        for neighbour, distance in graph.get_neighbours(node).items():
            if not G.has_edge(node, neighbour) and not G.has_edge(neighbour, node):
                G.add_edge(node, neighbour, weight=distance)

    # Plot nodes with labels
    pos = nx.get_node_attributes(G, 'pos')
    # pos = {node: np.array(coord) * 100000 for node, coord in pos.items()}

    # Get node values (you need to provide a way to get the value associated with each node)
    if graph.budget < 40:
        node_values = {node: graph.get_node(node).value*(285/1.5) for node in graph.get_nodes()}  # Assuming graph has `get_node_value` method
    else:
        node_values = {node: graph.get_node(node).value*(285) for node in graph.get_nodes()}
    # Map node values to colors
    node_colours = [get_node_colour(node_values[node]) for node in graph.get_nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours, node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Plot edges (the new part)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='grey', width=0.5)#, connectionstyle='arc3,rad=0.2', arrows=True)

    # Optionally, draw edge labels (distances)
    # edge_labels = {(n1, n2): f"{w['weight']:.1f}" for n1, n2, w in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    ax.margins(0)  # No margins around the graph

    # Make sure the plot fills the entire window
    plt.tight_layout()

    print("Setup graph")
    plt.savefig("Orienteering_Problem_Graph")
    plt.pause(0.0000001)
    return fig, ax, G, pos

# Update the plot with newly added edges
def update_plot(ax, G, pos, parent_idx, child_idx):
    G.add_edge(parent_idx, child_idx)
    ax.clear()
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, edge_color='blue')
    plt.draw()
    plt.pause(0.0000001)  # Small pause to allow the plot to update

def plot_final_path(ax, G, pos, graph: OrienteeringGraph, path, filename):
    red_edges = []
    edge_labels = {}

    for i in range(len(path) - 1):
        parent_idx = path[i]
        child_idx = path[i + 1]
        red_edges.append((parent_idx, child_idx))
        distance = graph.get_neighbours(parent_idx)[child_idx]
        edge_labels[(parent_idx, child_idx)] = f"{distance:.1f}"

    # Draw the red edges for the final path
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', width=2.0, ax=ax)

    # Draw edge labels for the budget/distance
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
    plt.savefig(filename)
    plt.draw()
    plt.pause(10)

def plot_rewards(rewards_log, filename, step=1000):
    sampled_rewards = rewards_log[::step]
    sampled_iterations = list(range(0, len(rewards_log), step))

    plt.figure()
    plt.plot(sampled_iterations, sampled_rewards, color="blue", linestyle="-", markersize=4, linewidth=1)
    plt.xlabel("Number of Rollouts")
    plt.ylabel("Reward")
    plt.title("Reward over Rollouts")
    plt.ylim(bottom=0)  # Start y-axis from zero
    plt.xlim(left=0)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_rewards_parallel(thread_rewards, averaged_rewards, filename, step=1000):
    plt.figure()

    sampled_rewards = averaged_rewards[::step]
    sampled_iterations = list(range(0, len(averaged_rewards), step))

    colors = ['red', 'green', 'orange', 'purple']
    for index, rewards in enumerate(thread_rewards):
        # Sample rewards for current thread
        sampled_rewards = rewards[::step]
        plt.plot(sampled_iterations, sampled_rewards, color=colors[index], linestyle="-", markersize=4, linewidth=1, label=f'Thread {index + 1}')

    sampled_averaged_rewards = averaged_rewards[::step]
    plt.plot(sampled_iterations, sampled_averaged_rewards, color='blue', linestyle="-", markersize=4, linewidth=2, label='Average Reward')
    
    plt.xlabel("Simulation Iterations")
    plt.ylabel("Reward")
    plt.title("Reward over Iterations")
    plt.ylim(bottom=0)  # Start y-axis from zero
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend()

    plt.savefig(filename)
    plt.show()

def plot_rewards_root(thread_rewards, filename, step=1000):
    #For each list within thread_rewards, get value of index and add to new array
    rollout_rewards = []
    
    # Loop through each index in the sublists
    for i in range(len(thread_rewards[0])):  
        for rewards_list in thread_rewards:
            rollout_rewards.append(rewards_list[i])

    plt.figure()

    sampled_rewards = rollout_rewards[::step]
    sampled_iterations = list(range(0, len(rollout_rewards), step))

    plt.figure()
    plt.plot(sampled_iterations, sampled_rewards, color="blue", linestyle="-", markersize=4, linewidth=1)
    plt.xlabel("Number of Rollouts")
    plt.ylabel("Reward")
    plt.title("Reward over Rollouts")
    plt.ylim(bottom=0)  # Start y-axis from zero
    plt.xlim(left=0)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()