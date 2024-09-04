class Node:
    def __init__(self, x, y, value):
        self.pos = (x,y)
        self.value = value
        self.neighbors = []
    
    def add_neighbor(self, neighbor, distance):
        self.neighbors.append((neighbor, distance))
