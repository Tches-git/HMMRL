import numpy as np
import cv2
import torch
from hmmlearn import hmm
from .graph import Graph
from .contour_processor import ContourProcessor
from .rl_optimizer import RLOptimizer

class GraphBuilder:
    def __init__(self, device: torch.device):
        self.graph = Graph(device=device)
        self._contour_processor = ContourProcessor(device=device)
        self.rl_optimizer = None
        self.device = device

    def build_from_images(self, image_paths, texts=None, bert_model=None, bert_tokenizer=None):
        if texts and len(texts) != len(image_paths):
            raise ValueError("文本数量必须与图像数量匹配")
        
        node_ids = self._initialize_nodes(image_paths, texts, bert_model, bert_tokenizer)
        enhanced_features = self._compute_enhanced_features(node_ids)
        optimal_sequence = self._train_hmm_and_get_sequence(enhanced_features, len(image_paths))
        stitched_image = self._stitch_images_based_on_sequence(node_ids, optimal_sequence)
        
        return self.graph, stitched_image

    def build_from_images_with_rl(self, image_paths, texts=None, bert_model=None, bert_tokenizer=None,
                                max_iterations=10, epsilon=0.1):
        if texts and len(texts) != len(image_paths):
            raise ValueError("文本数量必须与图像数量匹配")
        
        node_ids = self._initialize_nodes(image_paths, texts, bert_model, bert_tokenizer)
        enhanced_features = self._compute_enhanced_features(node_ids)
        n_states = len(node_ids)
        
        self.rl_optimizer = RLOptimizer(n_states, device=self.device)
        
        best_score = float('-inf')
        best_sequence = None
        best_image = None
        
        for iteration in range(max_iterations):
            print(f"RL优化迭代 {iteration + 1}/{max_iterations}")
            
            try:
                current_transition_matrix = self.rl_optimizer.get_transition_matrix()
                current_sequence = self._train_hmm_and_get_sequence(
                    enhanced_features, 
                    n_states,
                    transition_matrix=current_transition_matrix
                )
                current_image = self._stitch_images_based_on_sequence(node_ids, current_sequence)
                sorted_nodes = [self.graph.nodes[node_ids[i]] for i in np.argsort(current_sequence)]
                quality_score = self._evaluate_stitching_quality(current_image, sorted_nodes)
                
                print(f"当前迭代得分: {quality_score:.4f}")
                
                for i in range(len(current_sequence)-1):
                    state = current_sequence[i]
                    next_state = current_sequence[i+1]
                    reward = quality_score
                    self.rl_optimizer.update(state, next_state, reward, next_state)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_sequence = current_sequence.copy()
                    best_image = current_image.copy()
                    print(f"发现更好的解决方案，得分: {best_score:.4f}")
                
            except Exception as e:
                print(f"迭代 {iteration + 1} 出错: {str(e)}")
                continue
            
            if iteration > 5 and quality_score < best_score * 0.95:
                print("连续5次未改善，提前终止优化")
                break
        
        if best_image is None:
            print("强化学习优化失败，使用基础HMM方法")
            return self.build_from_images(image_paths, texts, bert_model, bert_tokenizer)
        
        print(f"优化完成，最终得分: {best_score:.4f}")
        return self.graph, best_image

    def _initialize_nodes(self, image_paths, texts, bert_model, bert_tokenizer):
        node_ids = []
        for img_idx, image_path in enumerate(image_paths):
            try:
                print(f"处理图像 {image_path}")
                
                original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if original_img is None:
                    raise ValueError(f"无法读取图像: {image_path}")
                
                text = texts[img_idx] if texts else None
                segments, chain_codes = self._contour_processor.process_image(image_path)
                
                contour_centroids = [self._compute_centroid(seg) for seg in segments]
                all_contour_points = np.vstack(segments)
                centroid = self._compute_centroid(all_contour_points)
                
                freeman_codes = {
                    'E1': chain_codes[0],
                    'E2': chain_codes[1],
                    'E3': chain_codes[2],
                    'E4': chain_codes[3]
                }
                
                node_id = self.graph.add_node(
                    freeman_codes=freeman_codes,
                    centroid=centroid,
                    contour_centroids=contour_centroids,
                    text=text,
                    bert_model=bert_model,
                    bert_tokenizer=bert_tokenizer,
                    contours=segments,
                    image_id=img_idx,
                    original_image=original_img
                )
                node_ids.append(node_id)
                
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {str(e)}")
                raise
                
        return node_ids

    def _compute_centroid(self, contour):
        M = cv2.moments(contour)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        return (cx, cy)

    def _compute_enhanced_features(self, node_ids):
        n_nodes = len(node_ids)
        features = torch.zeros((n_nodes, 8), device=self.device)
        
        for i, node1_id in enumerate(node_ids):
            node1 = self.graph.nodes[node1_id]
            shape_scores = torch.zeros(4, device=self.device)
            geometric_scores = torch.zeros(4, device=self.device)
            
            for j, node2_id in enumerate(node_ids):
                if i == j:
                    continue
                
                node2 = self.graph.nodes[node2_id]
                
                for edge_idx in range(4):
                    contour1 = node1.contours[edge_idx]
                    contour2 = node2.contours[(edge_idx + 2) % 4]
                    
                    shape_score = self._contour_processor.evaluate_contour_match(contour1, contour2)
                    geo_score = self._compute_geometric_compatibility(contour1, contour2)
                    
                    shape_scores[edge_idx] = max(shape_scores[edge_idx], torch.tensor(shape_score, device=self.device))
                    geometric_scores[edge_idx] = max(geometric_scores[edge_idx], torch.tensor(geo_score, device=self.device))
            
            features[i] = torch.cat([shape_scores, geometric_scores])
        
        return features.cpu().numpy()  # 返回 CPU 上的 NumPy 数组供 hmmlearn 使用

    def _compute_geometric_compatibility(self, contour1, contour2):
        try:
            len1 = cv2.arcLength(contour1, False)
            len2 = cv2.arcLength(contour2, False)
            
            length_ratio = min(len1, len2) / max(len1, len2)
            
            _, (_, _), angle1 = cv2.fitEllipse(contour1)
            _, (_, _), angle2 = cv2.fitEllipse(contour2)
            angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2)) / 180.0
            
            geometric_score = 0.7 * length_ratio + 0.3 * (1 - angle_diff)
            return geometric_score
            
        except Exception as e:
            print(f"几何特征计算错误: {str(e)}")
            return 0.0

    def _train_hmm_and_get_sequence(self, observations, n_states, transition_matrix=None):
        try:
            n_components = min(n_states, len(observations))
            
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=100,
                init_params="mc" if transition_matrix is None else ""
            )
            
            if transition_matrix is not None:
                if transition_matrix.shape != (n_components, n_components):
                    adjusted_matrix = np.zeros((n_components, n_components))
                    min_dim = min(transition_matrix.shape[0], n_components)
                    adjusted_matrix[:min_dim, :min_dim] = transition_matrix[:min_dim, :min_dim]
                    row_sums = adjusted_matrix.sum(axis=1)
                    row_sums[row_sums == 0] = 1
                    adjusted_matrix = adjusted_matrix / row_sums[:, np.newaxis]
                    model.transmat_ = adjusted_matrix
                else:
                    model.transmat_ = transition_matrix
            else:
                model.transmat_ = self._initialize_transition_matrix(observations, n_components)
            
            model.startprob_ = self._initialize_start_probabilities(observations, n_components)
            
            n_features = observations.shape[1]
            model.means_ = np.zeros((n_components, n_features))
            model.covars_ = np.ones((n_components, n_features))
            
            model.fit(observations)
            _, state_sequence = model.decode(observations, algorithm="viterbi")
            
            return state_sequence
            
        except Exception as e:
            print(f"HMM训练错误: {str(e)}")
            return np.arange(len(observations))

    def _initialize_start_probabilities(self, observations, n_components):
        start_probs = np.zeros(n_components)
        
        for i in range(n_components):
            edge_features = observations[i, :4]
            boundary_score = np.mean(edge_features)
            start_probs[i] = 1.0 - boundary_score
        
        start_probs = np.maximum(start_probs, 1e-5)
        start_probs /= start_probs.sum()
        
        return start_probs

    def _initialize_transition_matrix(self, observations, n_components):
        transmat = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(n_components):
                if i != j:
                    if i < len(observations) and j < len(observations):
                        transmat[i,j] = self._compute_transition_probability(
                            observations[i],
                            observations[j]
                        )
        
        np.fill_diagonal(transmat, 0.01)
        
        row_sums = transmat.sum(axis=1)
        row_sums[row_sums == 0] = 1
        transmat = transmat / row_sums[:, np.newaxis]
        
        return transmat

    def _compute_transition_probability(self, obs1, obs2):
        shape_scores1 = obs1[:4]
        geo_scores1 = obs1[4:8]
        shape_scores2 = obs2[:4]
        geo_scores2 = obs2[4:8]
        
        edge_compatibility = self._compute_edge_compatibility(
            shape_scores1, shape_scores2,
            geo_scores1, geo_scores2
        )
        
        spatial_constraint = self._compute_spatial_constraint(
            geo_scores1, geo_scores2
        )
        
        direction_consistency = self._compute_direction_consistency(
            geo_scores1, geo_scores2
        )
        
        return (0.4 * edge_compatibility +
                0.3 * spatial_constraint +
                0.3 * direction_consistency)

    def _compute_edge_compatibility(self, shape_scores1, shape_scores2, geo_scores1, geo_scores2):
        edge_scores = []
        for i in range(4):
            opposite_edge = (i + 2) % 4
            match_score = (
                0.7 * (shape_scores1[i] * shape_scores2[opposite_edge]) +
                0.3 * (geo_scores1[i] * geo_scores2[opposite_edge])
            )
            edge_scores.append(match_score)
        
        return max(edge_scores)

    def _compute_spatial_constraint(self, geo_scores1, geo_scores2):
        length_compatibility = np.minimum(geo_scores1, geo_scores2).mean()
        return length_compatibility

    def _compute_direction_consistency(self, geo_scores1, geo_scores2):
        direction_diff = np.abs(geo_scores1 - geo_scores2)
        consistency = 1.0 - np.mean(direction_diff)
        return max(0.0, consistency)

    def _evaluate_stitching_quality(self, stitched_image, sorted_nodes):
        edge_scores = []
        for i in range(len(sorted_nodes)-1):
            curr_node = sorted_nodes[i]
            next_node = sorted_nodes[i+1]
            edge_score = self._evaluate_edge_continuity(curr_node, next_node)
            edge_scores.append(edge_score)
        
        edge_quality = np.mean(edge_scores) if edge_scores else 0.0
        shape_score = self._evaluate_global_shape(stitched_image)
        texture_score = self._evaluate_texture_continuity(stitched_image)
        
        return (0.4 * edge_quality + 
                0.3 * shape_score + 
                0.3 * texture_score)

    def _evaluate_edge_continuity(self, node1, node2):
        best_score = 0
        for edge_idx in range(4):
            contour1 = node1.contours[edge_idx]
            contour2 = node2.contours[(edge_idx + 2) % 4]
            score = self._contour_processor.evaluate_contour_match(contour1, contour2)
            best_score = max(best_score, score)
        return best_score

    def _evaluate_global_shape(self, stitched_image):
        try:
            if len(stitched_image.shape) == 3:
                gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = stitched_image
                
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
                
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            if perimeter == 0:
                return 0.0
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return circularity
            
        except Exception as e:
            print(f"形状评估错误: {str(e)}")
            return 0.0

    def _evaluate_texture_continuity(self, stitched_image):
        try:
            if len(stitched_image.shape) == 3:
                gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = stitched_image
                
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            consistency = 1.0 - np.mean(gradient_magnitude) / 255.0
            return consistency
            
        except Exception as e:
            print(f"纹理评估错误: {str(e)}")
            return 0.0

    def _stitch_images_based_on_sequence(self, node_ids, sequence):
        if not node_ids or not sequence:
            print("错误: node_ids 或 sequence 为空")
            return np.zeros((100, 100), dtype=np.uint8)  # 返回空画布

        try:
            sorted_indices = np.argsort(sequence)
            sorted_nodes = [self.graph.nodes[node_ids[i]] for i in sorted_indices]
            
            max_canvas_size = 5000
            used_positions = np.zeros((max_canvas_size, max_canvas_size), dtype=bool)
            offsets = [(max_canvas_size//4, max_canvas_size//4)]
            
            # 处理第一个节点
            first_node = sorted_nodes[0]
            first_mask = np.zeros_like(first_node.original_image)
            cv2.drawContours(first_mask, [np.vstack(first_node.contours)], -1, (255), thickness=cv2.FILLED)
            h, w = first_mask.shape
            x_start, y_start = offsets[0]
            used_positions[x_start:x_start+h, y_start:y_start+w] = (first_mask > 0)
            
            # 处理后续节点
            for i in range(1, len(sorted_nodes)):
                prev_node = sorted_nodes[i-1]
                curr_node = sorted_nodes[i]
                
                best_edge = 0
                best_score = 0
                best_offset = None
                max_score = float('-inf')
                
                for edge_idx in range(4):
                    contour1 = prev_node.contours[edge_idx]
                    contour2 = curr_node.contours[(edge_idx + 2) % 4]
                    score = self._contour_processor.evaluate_contour_match(contour1, contour2)
                    
                    if score > best_score:
                        prev_offset = offsets[-1]
                        contour_length = self._get_contour_length(prev_node.contours[edge_idx])
                        
                        if edge_idx == 0:
                            new_offset = (prev_offset[0], prev_offset[1] + contour_length)
                        elif edge_idx == 1:
                            new_offset = (prev_offset[0] + contour_length, prev_offset[1])
                        elif edge_idx == 2:
                            new_offset = (prev_offset[0], prev_offset[1] - contour_length)
                        else:
                            new_offset = (prev_offset[0] - contour_length, prev_offset[1])
                        
                        x_pos, y_pos = new_offset
                        curr_mask = np.zeros_like(curr_node.original_image)
                        cv2.drawContours(curr_mask, [np.vstack(curr_node.contours)], -1, (255), thickness=cv2.FILLED)
                        h, w = curr_mask.shape
                        
                        if (0 <= x_pos < max_canvas_size-h and 
                            0 <= y_pos < max_canvas_size-w):
                            region = used_positions[x_pos:x_pos+h, y_pos:y_pos+w]
                            if not np.any(region & (curr_mask > 0)):
                                if score > max_score:
                                    max_score = score
                                    best_score = score
                                    best_edge = edge_idx
                                    best_offset = new_offset
                
                if best_offset is None:
                    best_offset = self._find_nearest_valid_position(
                        curr_node, offsets[-1], used_positions, max_canvas_size
                    )
                
                x_pos, y_pos = best_offset
                curr_mask = np.zeros_like(curr_node.original_image)
                cv2.drawContours(curr_mask, [np.vstack(curr_node.contours)], -1, (255), thickness=cv2.FILLED)
                h, w = curr_mask.shape
                used_positions[x_pos:x_pos+h, y_pos:y_pos+w] |= (curr_mask > 0)
                offsets.append(best_offset)
            
            # 计算画布尺寸
            if len(offsets) < 1:
                print("错误: offsets 列表为空")
                return np.zeros((100, 100), dtype=np.uint8)
            
            min_x = min(offset[0] for offset in offsets)
            min_y = min(offset[1] for offset in offsets)
            max_x = max(offset[0] + node.original_image.shape[0] for offset, node in zip(offsets, sorted_nodes))
            max_y = max(offset[1] + node.original_image.shape[1] for offset, node in zip(offsets, sorted_nodes))
            
            canvas_width = int(max_y - min_y)
            canvas_height = int(max_x - min_x)
            
            if canvas_width <= 0 or canvas_height <= 0:
                print("错误: 画布尺寸无效")
                return np.zeros((100, 100), dtype=np.uint8)
            
            stitched = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
            
            # 将节点图像拼接至画布
            for node, offset in zip(sorted_nodes, offsets):
                x_pos = int(offset[0] - min_x)
                y_pos = int(offset[1] - min_y)
                
                mask = np.zeros_like(node.original_image)
                cv2.drawContours(mask, [np.vstack(node.contours)], -1, (255), thickness=cv2.FILLED)
                fragment = cv2.bitwise_and(node.original_image, node.original_image, mask=mask)
                
                h, w = fragment.shape
                x_end = min(x_pos + h, canvas_height)
                y_end = min(y_pos + w, canvas_width)
                
                if x_pos >= 0 and y_pos >= 0 and x_end <= canvas_height and y_end <= canvas_width:
                    stitched[x_pos:x_end, y_pos:y_end] = fragment[:x_end-x_pos, :y_end-y_pos]
            
            return stitched

        except Exception as e:
            print(f"图像拼接错误: {str(e)}")
            return np.zeros((100, 100), dtype=np.uint8)

    def _find_nearest_valid_position(self, node, base_offset, used_positions, max_size):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x_base, y_base = base_offset
        
        for distance in range(1, 100):
            for dx, dy in directions:
                x_new = x_base + dx * distance
                y_new = y_base + dy * distance
                
                mask = np.zeros_like(node.original_image)
                cv2.drawContours(mask, [np.vstack(node.contours)], -1, (255), thickness=cv2.FILLED)
                h, w = mask.shape
                
                if (0 <= x_new < max_size-h and 
                    0 <= y_new < max_size-w):
                    
                    region = used_positions[x_new:x_new+h, y_new:y_new+w]
                    if not np.any(region & (mask > 0)):
                        return (x_new, y_new)
        
        return base_offset

    def _get_contour_length(self, contour):
        return int(cv2.arcLength(contour, False))