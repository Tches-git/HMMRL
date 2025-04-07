import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用之前定义的类结构
class GraphNode:
    def __init__(
        self,
        node_id: int,
        freeman_codes: Dict[str, List[int]],
        centroid: Tuple[float, float],
        contour_centroids: List[Tuple[float, float]],
        semantic_vector: Optional[np.ndarray] = None,
        contours: Optional[List[np.ndarray]] = None,
        image_id: Optional[int] = None
    ):
        self.id = node_id
        self.freeman_codes = freeman_codes
        self.centroid = centroid
        self.contour_centroids = contour_centroids
        self.semantic_vector = semantic_vector
        self.contours = contours
        self.image_id = image_id
        
    def __repr__(self):
        return f"GraphNode(id={self.id}, image_id={self.image_id}, centroid={self.centroid})"

class GraphEdge:
    def __init__(self, node1: int, node2: int, weight: float):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        
    def __repr__(self):
        return f"GraphEdge({self.node1} <-> {self.node2}, weight={self.weight:.4f})"

class Graph:
    def __init__(self):
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._next_node_id = 0
        
    def add_node(
        self,
        freeman_codes: Dict[str, List[int]],
        centroid: Tuple[float, float],
        contour_centroids: List[Tuple[float, float]],
        text: Optional[str] = None,
        bert_model: Optional[BertModel] = None,
        bert_tokenizer: Optional[BertTokenizer] = None,
        contours: Optional[List[np.ndarray]] = None,
        image_id: Optional[int] = None
    ) -> int:
        semantic_vector = None
        if text is not None and bert_model is not None and bert_tokenizer is not None:
            semantic_vector = self._generate_bert_vector(text, bert_model, bert_tokenizer)
        
        node_id = self._next_node_id
        self.nodes[node_id] = GraphNode(
            node_id=node_id,
            freeman_codes=freeman_codes,
            centroid=centroid,
            contour_centroids=contour_centroids,
            semantic_vector=semantic_vector,
            contours=contours,
            image_id=image_id
        )
        self._next_node_id += 1
        return node_id
    
    def add_edge(self, node1: int, node2: int, weight: float):
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("One or both nodes do not exist in the graph")
        self.edges.append(GraphEdge(node1, node2, weight))
    
    def _generate_bert_vector(
        self,
        text: str,
        bert_model: BertModel,
        bert_tokenizer: BertTokenizer
    ) -> np.ndarray:
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

class GraphBuilder:
    def __init__(self):
        self.graph = Graph()
        self._contour_processor = ContourProcessor()
    
    def build_from_images(
        self,
        image_paths: List[str],
        texts: Optional[List[str]] = None,
        bert_model: Optional[BertModel] = None,
        bert_tokenizer: Optional[BertTokenizer] = None
    ) -> 'Graph':
        if texts is not None and len(texts) != len(image_paths):
            raise ValueError("文本数量必须与图像数量匹配")
        
        for img_idx, image_path in enumerate(image_paths):
            text = texts[img_idx] if texts is not None else None
            segments, chain_codes = self._contour_processor.process_image(image_path)
            
            contour_centroids = []
            for seg in segments:
                M = cv2.moments(seg)
                cx = int(M['m10'] / (M['m00'] + 1e-5))
                cy = int(M['m01'] / (M['m00'] + 1e-5))
                contour_centroids.append((cx, cy))
            
            all_contour_points = np.vstack(segments)
            M = cv2.moments(all_contour_points)
            centroid_x = int(M['m10'] / (M['m00'] + 1e-5))
            centroid_y = int(M['m01'] / (M['m00'] + 1e-5))
            centroid = (centroid_x, centroid_y)
            
            freeman_codes = {
                'E1': chain_codes[0],
                'E2': chain_codes[1],
                'E3': chain_codes[2],
                'E4': chain_codes[3]
            }
            
            self.graph.add_node(
                freeman_codes=freeman_codes,
                centroid=centroid,
                contour_centroids=contour_centroids,
                text=text,
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                contours=segments,
                image_id=img_idx
            )
        
        return self.graph

class ContourProcessor:
    @staticmethod
    def freeman_chain_code(contour, closed=True):
        directions = [0, 1, 2, 3, 4, 5, 6, 7]
        dx = [1, 1, 0, -1, -1, -1, 0, 1]
        dy = [0, 1, 1, 1, 0, -1, -1, -1]
        chain_code = []
        n = len(contour)
        
        for i in range(n - 1 if not closed else n):
            current = contour[i % n][0]
            next_p = contour[(i + 1) % n][0]
            delta = (next_p[0] - current[0], next_p[1] - current[1])
            for j, (ddx, ddy) in enumerate(zip(dx, dy)):
                if delta == (ddx, ddy):
                    chain_code.append(j)
                    break
            else:
                raise ValueError(f"Invalid direction delta: {delta} at index {i}")
        return chain_code
    
    @staticmethod
    def preprocess(img):
        blurred = cv2.GaussianBlur(img, (7, 7), 1.5)
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 6)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        return closed
    
    @staticmethod
    def is_valid_contour(cnt, img_shape):
        area = cv2.contourArea(cnt)
        if area < 500:
            return False
        x, y, w, h = cv2.boundingRect(cnt)
        if (x == 0 or y == 0 or x + w >= img_shape[1] - 1 or y + h >= img_shape[0] - 1):
            return False
        solidity = area / (w * h)
        return 0.3 < solidity < 0.9
    
    def process_image(self, image_path: str):
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"无法读取图像: {image_path}")
        processed = self.preprocess(original)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_contours = [cnt for cnt in contours if self.is_valid_contour(cnt, original.shape)]
        
        if not valid_contours:
            raise ValueError("未找到有效轮廓，请调整预处理参数")
        
        best_contour = max(valid_contours, key=lambda c: cv2.contourArea(c))
        perimeter = cv2.arcLength(best_contour, True)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(best_contour, epsilon, True)
        
        for _ in range(20):
            if len(approx) == 4:
                break
            elif len(approx) > 4:
                epsilon *= 1.1
            else:
                epsilon *= 0.9
            approx = cv2.approxPolyDP(best_contour, epsilon, True)
        else:
            raise ValueError("无法近似为恰好4个点")
        
        contour_points = best_contour.reshape(-1, 2)
        indices = []
        for p in approx.reshape(-1, 2):
            distances = np.sum((contour_points - p)**2, axis=1)
            idx = np.argmin(distances)
            indices.append(idx)
        indices.sort()
        
        if len(set(indices)) != 4:
            raise ValueError("近似点映射到重复索引")
        
        seg1 = best_contour[indices[0]:indices[1]+1]
        seg2 = best_contour[indices[1]:indices[2]+1]
        seg3 = best_contour[indices[2]:indices[3]+1]
        seg4 = np.vstack((best_contour[indices[3]:], best_contour[:indices[0]+1]))
        
        segments = [seg1, seg2, seg3, seg4]
        chain_codes = [self.freeman_chain_code(seg, closed=False) for seg in segments]
        return segments, chain_codes

# 新增匹配类
class GraphMatcher:
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def compute_chain_similarity(self, chain1: List[int], chain2: List[int]) -> float:
        """计算两个链码的相似性，使用余弦相似度"""
        len1, len2 = len(chain1), len(chain2)
        if len1 == 0 or len2 == 0:
            return 0.0
        max_len = max(len1, len2)
        chain1_padded = chain1 + [0] * (max_len - len1)
        chain2_padded = chain2 + [0] * (max_len - len2)
        similarity = 1 - cosine(chain1_padded, chain2_padded)
        return similarity if similarity > 0 else 0.0
    
    def match_and_update_edges(self, similarity_threshold: float = 0.7) -> Graph:
        """比对轮廓链码，调整图的边，保留拼接可能性最高的边"""
        self.graph.edges = []  # 清空现有边
        nodes = list(self.graph.nodes.values())
        n = len(nodes)
        
        for i in range(n):
            for j in range(i + 1, n):
                node1, node2 = nodes[i], nodes[j]
                best_similarity = 0.0
                best_edge_info = None
                
                # 比较每对链码
                for e1_key, chain1 in node1.freeman_codes.items():
                    for e2_key, chain2 in node2.freeman_codes.items():
                        similarity = self.compute_chain_similarity(chain1, chain2)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_edge_info = (node1.id, e1_key, node2.id, e2_key)
                
                # 如果最佳相似性超过阈值，添加边
                if best_similarity >= similarity_threshold:
                    self.graph.add_edge(best_edge_info[0], best_edge_info[2], best_similarity)
                    print(f"添加边: Node {best_edge_info[0]} ({best_edge_info[1]}) <-> Node {best_edge_info[2]} ({best_edge_info[3]}), 相似性={best_similarity:.4f}")
        
        return self.graph
    
    def visualize(self, original_imgs: Dict[int, np.ndarray]):
        plt.figure(figsize=(15, 8))
        
        unique_image_ids = set(node.image_id for node in self.graph.nodes.values())
        for idx, img_id in enumerate(unique_image_ids, 1):
            vis_img = cv2.cvtColor(original_imgs[img_id], cv2.COLOR_GRAY2BGR)
            node = next(n for n in self.graph.nodes.values() if n.image_id == img_id)
            if node.contours is not None:
                for i, contour in enumerate(node.contours):
                    color = (0, 255, 0) if i == 0 else \
                            (255, 0, 0) if i == 1 else \
                            (0, 0, 255) if i == 2 else \
                            (255, 255, 0)
                    cv2.polylines(vis_img, [contour], False, color, 2)
                    cv2.circle(vis_img, (int(node.contour_centroids[i][0]), 
                                      int(node.contour_centroids[i][1])), 5, (255, 255, 255), -1)
                cv2.circle(vis_img, (int(node.centroid[0]), int(node.centroid[1])), 7, (0, 0, 255), -1)
            
            plt.subplot(len(unique_image_ids), 2, 2 * (idx - 1) + 1)
            plt.imshow(vis_img)
            plt.title(f'图像 {img_id} 图结构')
        
        plt.subplot(1, 2, 2)
        for node in self.graph.nodes.values():
            plt.scatter(node.centroid[0], node.centroid[1], s=100, label=f'Node {node.id} (Img {node.image_id})')
        for edge in self.graph.edges:
            node1 = self.graph.nodes[edge.node1]
            node2 = self.graph.nodes[edge.node2]
            plt.plot([node1.centroid[0], node2.centroid[0]], 
                     [node1.centroid[1], node2.centroid[1]], 'k-', 
                     label=f'w={edge.weight:.2f}' if edge.weight > 0 else None)
        plt.title('最佳拼接图拓扑')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# 示例用法
if __name__ == "__main__":
    bert_model = None
    bert_tokenizer = None
    
    builder = GraphBuilder()
    image_paths = [
        r"C:\Users\28489\Desktop\paired\31\1.jpg",
        r"C:\Users\28489\Desktop\paired\31\2.jpg",
        r"C:\Users\28489\Desktop\paired\31\4.jpg",
        r"C:\Users\28489\Desktop\paired\31\5.jpg",
        r"C:\Users\28489\Desktop\paired\31\6.jpg"    
    ]
    texts = ["图像1描述", "图像2描述", "图像4描述", "图像5描述", "图像6描述"]
    
    try:
        # 构建初始图（仅节点）
        graph = builder.build_from_images(
            image_paths=image_paths,
            texts=texts,
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer
        )
        print(f"初始图包含 {len(graph.nodes)} 个节点和 {len(graph.edges)} 条边")
        
        # 匹配并调整边
        matcher = GraphMatcher(graph)
        matched_graph = matcher.match_and_update_edges(similarity_threshold=0.7)
        print(f"匹配后图包含 {len(matched_graph.nodes)} 个节点和 {len(matched_graph.edges)} 条边")
        
        # 可视化
        original_imgs = {i: cv2.imread(path, cv2.IMREAD_GRAYSCALE) for i, path in enumerate(image_paths)}
        matcher.visualize(original_imgs)
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")