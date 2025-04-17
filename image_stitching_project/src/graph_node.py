from typing import Dict, List, Optional, Tuple
import numpy as np

class GraphNode:
    def __init__(
        self,
        node_id: int,
        freeman_codes: Dict[str, List[int]],
        centroid: Tuple[float, float],
        contour_centroids: List[Tuple[float, float]],
        semantic_vector: Optional[np.ndarray] = None,
        contours: Optional[List[np.ndarray]] = None,
        image_id: Optional[int] = None,
        original_image: Optional[np.ndarray] = None
    ):
        self.id = node_id
        self.freeman_codes = freeman_codes
        self.centroid = centroid
        self.contour_centroids = contour_centroids
        self.semantic_vector = semantic_vector
        self.contours = contours
        self.image_id = image_id
        self.original_image = original_image
        
    def __repr__(self):
        return f"GraphNode(id={self.id}, image_id={self.image_id}, centroid={self.centroid})"