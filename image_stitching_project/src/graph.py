from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from .graph_node import GraphNode
from .graph_edge import GraphEdge

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Graph:
    def __init__(self, device: torch.device):
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._next_node_id = 0
        self.device = device
        
    def add_node(self, freeman_codes, centroid, contour_centroids, text=None, bert_model=None, bert_tokenizer=None, contours=None, image_id=None, original_image=None):
        semantic_vector = None
        if text and bert_model and bert_tokenizer:
            semantic_vector = self._generate_bert_vector(text, bert_model, bert_tokenizer)
        
        node_id = self._next_node_id
        self.nodes[node_id] = GraphNode(
            node_id=node_id,
            freeman_codes=freeman_codes,
            centroid=centroid,
            contour_centroids=contour_centroids,
            semantic_vector=semantic_vector,
            contours=contours,
            image_id=image_id,
            original_image=original_image
        )
        self._next_node_id += 1
        return node_id
    
    def add_edge(self, node1: int, node2: int, weight: Optional[float] = None):
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("One or both nodes do not exist in the graph")
        self.edges.append(GraphEdge(node1, node2, weight))
    
    def _generate_bert_vector(self, text, bert_model, bert_tokenizer):
        bert_model = bert_model.to(self.device)
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 返回 CPU 上的 NumPy 数组
    
    def visualize(self, stitched_image: Optional[np.ndarray] = None):
        plt.figure(figsize=(15, 10))
        
        unique_image_ids = set(node.image_id for node in self.nodes.values())
        for idx, img_id in enumerate(unique_image_ids, 1):
            node = next(n for n in self.nodes.values() if n.image_id == img_id)
            original_img = node.original_image
            
            mask = np.zeros_like(original_img)
            cv2.drawContours(mask, [np.vstack(node.contours)], -1, (255), thickness=cv2.FILLED)
            fragment = cv2.bitwise_and(original_img, original_img, mask=mask)
            
            plt.subplot(2, len(unique_image_ids), idx)
            plt.imshow(original_img, cmap='gray')
            plt.title(f'原始图像 {img_id}')
            plt.axis('off')
            
            plt.subplot(2, len(unique_image_ids), idx + len(unique_image_ids))
            plt.imshow(fragment, cmap='gray')
            plt.title(f'提取的残片 {img_id}')
            plt.axis('off')
        
        if stitched_image is not None:
            plt.figure(figsize=(10, 6))
            plt.imshow(stitched_image, cmap='gray')
            plt.title('拼接复原结果')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()