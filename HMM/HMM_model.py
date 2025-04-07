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
        freeman_codes: Dict[str, List[int]],  # 四个方向的Freeman编码
        centroid: Tuple[float, float],       # 质心位置 (x, y)
        semantic_vector: Optional[np.ndarray] = None,  # BERT语义向量
        contour_segment: Optional[np.ndarray] = None  # 原始轮廓段
    ):
        self.id = node_id
        self.freeman_codes = freeman_codes
        self.centroid = centroid
        self.semantic_vector = semantic_vector
        self.contour_segment = contour_segment
        
    def __repr__(self):
        return f"GraphNode(id={self.id}, centroid={self.centroid})"

class GraphEdge:
    def __init__(self, node1: int, node2: int, weight: Optional[float] = None):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        
    def __repr__(self):
        return f"GraphEdge({self.node1} <-> {self.node2}, weight={self.weight})"

class GraphBuilder:
    def __init__(self):
        self.graph = Graph()
        self._contour_processor = ContourProcessor()
    
    def build_from_image(
        self,
        image_path: str,
        text: Optional[str] = None,
        bert_model: Optional[BertModel] = None,
        bert_tokenizer: Optional[BertTokenizer] = None
    ) -> 'Graph':
        """从图像构建图结构"""
        # 处理图像获取轮廓特征
        segments, chain_codes = self._contour_processor.process_image(image_path)
        
        # 计算每个轮廓段的质心
        centroids = []
        for seg in segments:
            M = cv2.moments(seg)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
        
        # 添加节点到图中
        node_ids = []
        for i, (chain, centroid, seg) in enumerate(zip(chain_codes, centroids, segments)):
            freeman_codes = {
                'top': chain if i == 0 else [],
                'right': chain if i == 1 else [],
                'bottom': chain if i == 2 else [],
                'left': chain if i == 3 else []
            }
            node_id = self.graph.add_node(
                freeman_codes=freeman_codes,
                centroid=centroid,
                text=text,
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                contour_segment=seg
            )
            node_ids.append(node_id)
        
        # 自动添加边（连接相邻节点）
        for i in range(len(node_ids)):
            j = (i + 1) % len(node_ids)
            self.graph.add_edge(node_ids[i], node_ids[j])
        
        return self.graph

class Graph:
    def __init__(self):
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._next_node_id = 0
        
    def add_node(
        self,
        freeman_codes: Dict[str, List[int]],
        centroid: Tuple[float, float],
        text: Optional[str] = None,
        bert_model: Optional[BertModel] = None,
        bert_tokenizer: Optional[BertTokenizer] = None,
        contour_segment: Optional[np.ndarray] = None
    ) -> int:
        """添加节点到图中"""
        semantic_vector = None
        if text is not None and bert_model is not None and bert_tokenizer is not None:
            semantic_vector = self._generate_bert_vector(text, bert_model, bert_tokenizer)
        
        node_id = self._next_node_id
        self.nodes[node_id] = GraphNode(
            node_id=node_id,
            freeman_codes=freeman_codes,
            centroid=centroid,
            semantic_vector=semantic_vector,
            contour_segment=contour_segment
        )
        self._next_node_id += 1
        return node_id
    
    def add_edge(self, node1: int, node2: int, weight: Optional[float] = None):
        """添加边到图中"""
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("One or both nodes do not exist in the graph")
        self.edges.append(GraphEdge(node1, node2, weight))
    
    def _generate_bert_vector(
        self,
        text: str,
        bert_model: BertModel,
        bert_tokenizer: BertTokenizer
    ) -> np.ndarray:
        """使用BERT生成语义向量"""
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    def visualize(self, original_img: np.ndarray = None):
        """可视化图结构"""
        plt.figure(figsize=(15, 8))
        
        if original_img is not None:
            # 在原始图像上绘制图结构
            vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
            
            # 绘制节点轮廓
            for node in self.nodes.values():
                if node.contour_segment is not None:
                    color = (0, 255, 0) if node.id % 4 == 0 else \
                            (255, 0, 0) if node.id % 4 == 1 else \
                            (0, 0, 255) if node.id % 4 == 2 else \
                            (255, 255, 0)
                    cv2.polylines(vis_img, [node.contour_segment], False, color, 2)
                    cv2.circle(vis_img, (int(node.centroid[0]), int(node.centroid[1])), 5, (0, 0, 255), -1)
            
            # 绘制边
            for edge in self.edges:
                node1 = self.nodes[edge.node1]
                node2 = self.nodes[edge.node2]
                cv2.line(vis_img, 
                        (int(node1.centroid[0]), int(node1.centroid[1])),
                        (int(node2.centroid[0]), int(node2.centroid[1])),
                        (255, 255, 255), 2)
            
            plt.subplot(121)
            plt.imshow(vis_img)
            plt.title('图结构可视化')
        
        # 绘制图的关系图
        plt.subplot(122)
        for node in self.nodes.values():
            plt.scatter(node.centroid[0], node.centroid[1], s=100, label=f'Node {node.id}')
        for edge in self.edges:
            node1 = self.nodes[edge.node1]
            node2 = self.nodes[edge.node2]
            plt.plot([node1.centroid[0], node2.centroid[0]], 
                    [node1.centroid[1], node2.centroid[1]], 'k-')
        plt.title('图关系拓扑')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

class ContourProcessor:
    """轮廓处理类，封装所有轮廓相关操作"""
    
    @staticmethod
    def freeman_chain_code(contour, closed=True):
        """改进的Freeman链码生成函数"""
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
        """图像预处理"""
        blurred = cv2.GaussianBlur(img, (7, 7), 1.5)
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 6)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        return closed
    
    @staticmethod
    def is_valid_contour(cnt, img_shape):
        """轮廓有效性验证"""
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
        """处理图像并返回分割后的轮廓段和链码"""
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
    # 初始化BERT模型和分词器(可选)
    bert_model = None
    bert_tokenizer = None
    
    # 创建图构建器
    builder = GraphBuilder()
    
    # 从图像构建图
    try:
        graph = builder.build_from_image(
            image_path=r"C:\Users\28489\Desktop\paired\7\3.jpg",  # 替换为您的图像路径
            text="示例对象",  # 可选文本描述
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer
        )
        
        # 打印图信息
        print(f"构建的图包含 {len(graph.nodes)} 个节点和 {len(graph.edges)} 条边")
        
        # 可视化图结构
        original_img = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)
        graph.visualize(original_img)
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")