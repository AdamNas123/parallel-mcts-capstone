import math
from state import State

def mcts(graph, root, Tmax, max_iterations):
    for _ in range(max_iterations):
        node = select_node(root)
        if not is_terminal(node):
            children = expand(node, graph)
            for child in children:
                score = rollout(child, graph, Tmax)
                backpropagate(child, score)
    return best_path(root)


# Selection Phase - Select Child Nodes that maximise Upper Confidence Bound 1: 
def ucb1(total_score, visits, parent_visits, exploration_constant):
    return (total_score / visits) + exploration_constant * math.sqrt(2 * math.log(parent_visits) / visits)


# Expansion Phase - If selected node is not terminal, expand it by adding child node for each valid action
def expand(state, graph):
    children = []
    for neighbor, distance in graph[state.current_node].neighbors:
        if neighbor not in state.visited and (state.time_spent + distance <= Tmax):
            new_state = State(neighbor, state.visited + [neighbor], state.time_spent + distance, state.total_score + graph[neighbor].score)
            children.append(new_state)
    return children

# Rollout/Simulation Phase - Simulate outcome by randomly choosing valid actions until terminal state reached
import random

def rollout(state, graph, Tmax):
    current_state = state
    while current_state.time_spent < Tmax:
        possible_moves = [n for n, d in graph[current_state.current_node].neighbors if n not in current_state.visited and (current_state.time_spent + d <= Tmax)]
        if not possible_moves:
            break
        next_move = random.choice(possible_moves)
        current_state = State(next_move, current_state.visited + [next_move], current_state.time_spent + graph[current_state.current_node].neighbors[next_move], current_state.total_score + graph[next_move].score)
    return current_state.total_score

# Backpropagation Phase - After Simulation, propagate score back up tree. Updates total score and visit count for each node.
def backpropagate(node, score):
    while node is not None:
        node.total_score += score
        node.visits += 1
        node = node.parent
