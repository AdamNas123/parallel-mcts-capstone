from orienteering_problem import OrienteeringGraph
class MCTSNode:
    def __init__(self,node_index, parent=None, path=None):
        self.node_index = node_index
        self.parent = parent
        self.visits = 0
        self.value = 0
        self.children = []
        self.path = path if path is not None else [node_index]
    
    def add_child(self, child_node):
        self.children.append(child_node)

    def update_value(self, reward):
        self.value += reward
        self.visits += 1

    def get_path_distance(self, graph: OrienteeringGraph):
        distance = 0
        for i in range(len(self.path) - 1):
            distance += graph.get_distance(self.path[i], self.path[i + 1])
        return distance

