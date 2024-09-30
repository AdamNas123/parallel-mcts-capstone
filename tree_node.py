from orienteering_problem import OrienteeringGraph
class MCTSNode:
    def __init__(self,op_node_index, graph: OrienteeringGraph, parent=None, path=None, is_root=False):
        self.op_node_index = op_node_index
        self.parent = parent
        self.visits = 0
        self.children = []
        self.path = path if path is not None else [op_node_index]
        self.value = sum(graph.get_node(op_node).value for op_node in self.path)
        self.is_root = is_root  # New flag for root node
    
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

