from typing import Optional

class GraphEdge:
    def __init__(self, node1: int, node2: int, weight: Optional[float] = None):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        
    def __repr__(self):
        return f"GraphEdge({self.node1} <-> {self.node2}, weight={self.weight})"