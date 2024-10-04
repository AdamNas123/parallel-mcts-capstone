import matplotlib.pyplot as plt
from orienteering_problem import OrienteeringGraph
import networkx as nx
import numpy as np

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
        for neighbor, distance in graph.get_neighbours(node).items():
            G.add_edge(node, neighbor, weight=distance)

    # Plot nodes with labels
    pos = nx.get_node_attributes(G, 'pos')
    # pos = {node: np.array(coord) * 100000 for node, coord in pos.items()}
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500)

    # Plot edges (the new part)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=1.5)

    # Optionally, draw edge labels (distances)
    edge_labels = {(n1, n2): f"{w['weight']:.1f}" for n1, n2, w in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)


    print("Setup graph")
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
