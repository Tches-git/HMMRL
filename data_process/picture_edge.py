import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import BertModel, BertTokenizer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GraphNode:
    def __init__(
        self,
        node_id: int,
        freeman_codes: Dict[str, List[int]],  # E1-E4的Freeman链码
        centroid: Tuple[float, float],       # 整个残片的质心位置 (x, y)
        contour_centroids: List[Tuple[float, float]],  # 每个轮廓段的质心
        semantic_vector: Optional[np.ndarray] = None,  # BERT语义向量
        contours: Optional[List[np.ndarray]] = None,   # 所有轮廓段数据
        image_id: Optional[int] = None        # 图像ID
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
    def __init__(self, node1: int, node2: int, weight: Optional[float] = None):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        
    def __repr__(self):
        return f"GraphEdge({self.node1} <-> {self.node2}, weight={self.weight})"

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
    
    def add_edge(self, node1: int, node2: int, weight: Optional[float] = None):
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
    
    def visualize(self, original_imgs: Dict[int, np.ndarray] = None):
        plt.figure(figsize=(15, 8))
        
        if original_imgs is not None:
            unique_image_ids = set(node.image_id for node in self.nodes.values())
            for idx, img_id in enumerate(unique_image_ids, 1):
                vis_img = cv2.cvtColor(original_imgs[img_id], cv2.COLOR_GRAY2BGR)
                node = next(n for n in self.nodes.values() if n.image_id == img_id)
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
        for node in self.nodes.values():
            plt.scatter(node.centroid[0], node.centroid[1], s=100, label=f'Node {node.id} (Img {node.image_id})')
        for edge in self.edges:
            node1 = self.nodes[edge.node1]
            node2 = self.nodes[edge.node2]
            plt.plot([node1.centroid[0], node2.centroid[0]], 
                    [node1.centroid[1], node2.centroid[1]], 'k-')
        plt.title('整体图关系拓扑')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

class GraphBuilder:
    def __init__(self):
        self.graph = Graph()
        self._contour_processor = ContourProcessor()
    
    def build_from_images(
        self,
        image_paths: List[str],
        texts: Optional[List[str]] = None,
        bert_model: Optional[BertModel] = None,
        bert_tokenizer: Optional[BertTokenizer] = None,
        connect_all: bool = False  # 是否连接所有节点
    ) -> 'Graph':
        if texts is not None and len(texts) != len(image_paths):
            raise ValueError("文本数量必须与图像数量匹配")
        
        node_ids = []
        for img_idx, image_path in enumerate(image_paths):
            text = texts[img_idx] if texts is not None else None
            segments, chain_codes = self._contour_processor.process_image(image_path)
            
            # 计算每个轮廓段的质心
            contour_centroids = []
            for seg in segments:
                M = cv2.moments(seg)
                cx = int(M['m10'] / (M['m00'] + 1e-5))
                cy = int(M['m01'] / (M['m00'] + 1e-5))
                contour_centroids.append((cx, cy))
            
            # 计算整个残片的质心（基于所有轮廓点）
            all_contour_points = np.vstack(segments)
            M = cv2.moments(all_contour_points)
            centroid_x = int(M['m10'] / (M['m00'] + 1e-5))
            centroid_y = int(M['m01'] / (M['m00'] + 1e-5))
            centroid = (centroid_x, centroid_y)
            
            # 构建Freeman链码字典
            freeman_codes = {
                'E1': chain_codes[0],
                'E2': chain_codes[1],
                'E3': chain_codes[2],
                'E4': chain_codes[3]
            }
            
            # 添加节点
            node_id = self.graph.add_node(
                freeman_codes=freeman_codes,
                centroid=centroid,
                contour_centroids=contour_centroids,
                text=text,
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                contours=segments,
                image_id=img_idx
            )
            node_ids.append(node_id)
        
        # 可选：连接所有节点（形成完全图）
        if connect_all:
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    self.graph.add_edge(node_ids[i], node_ids[j])
        
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
        if (x == 0 or y == 0 or 
            x + w >= img_shape[1] - 1 or 
            y + h >= img_shape[0] - 1):
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

# 示例用法
if __name__ == "__main__":
    bert_model = None
    bert_tokenizer = None
    
    builder = GraphBuilder()
    
    image_paths = [
        r"C:\Users\28489\Desktop\paired\7\1.jpg",
        r"C:\Users\28489\Desktop\paired\7\2.jpg",
        r"C:\Users\28489\Desktop\paired\7\3.jpg"
    ]
    texts = ["图像1描述", "图像2描述", "图像3描述"]
    
    try:
        graph = builder.build_from_images(
            image_paths=image_paths,
            texts=texts,
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer,
            connect_all=True  # 可选：连接所有节点
        )
        
        print(f"构建的图包含 {len(graph.nodes)} 个节点和 {len(graph.edges)} 条边")
        
        original_imgs = {i: cv2.imread(path, cv2.IMREAD_GRAYSCALE) for i, path in enumerate(image_paths)}
        graph.visualize(original_imgs)
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")