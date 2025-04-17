import cv2
import numpy as np
from typing import List, Tuple

class ContourProcessor:
    def process_image(self, image_path: str) -> Tuple[List[np.ndarray], List[List[int]]]:
        try:
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                raise ValueError(f"无法读取图像: {image_path}")

            processed = self.preprocess(original)
            if processed is None:
                raise ValueError("图像预处理失败")
            
            contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                raise ValueError("未找到任何轮廓")
            
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
            
            if len(approx) != 4:
                raise ValueError("无法近似为恰好4个点")

            contour_points = best_contour.reshape(-1, 2)
            indices = [np.argmin(np.sum((contour_points - p)**2, axis=1)) 
                      for p in approx.reshape(-1, 2)]
            indices.sort()

            if len(set(indices)) != 4:
                raise ValueError("近似点映射到重复索引")

            segments = [
                best_contour[indices[0]:indices[1]+1],
                best_contour[indices[1]:indices[2]+1],
                best_contour[indices[2]:indices[3]+1],
                np.vstack((best_contour[indices[3]:], best_contour[:indices[0]+1]))
            ]

            chain_codes = [self.freeman_chain_code(seg, closed=False) for seg in segments]
            
            if len(segments) != 4 or len(chain_codes) != 4:
                raise ValueError("轮廓分割或链码生成失败")
            
            return segments, chain_codes

        except Exception as e:
            print(f"图像处理错误: {str(e)}")
            raise

    def preprocess(self, img):
        if img is None:
            return None
            
        blurred = cv2.GaussianBlur(img, (7, 7), 1.5)
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            21, 
            6
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        return closed

    def is_valid_contour(self, cnt, img_shape):
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

    def freeman_chain_code(self, contour, closed=True):
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

    def compute_shape_context(self, contour, n_points=100, n_bins_r=5, n_bins_theta=12):
        contour = contour.squeeze()
        sampled_points = np.array([
            contour[int(i * len(contour) / n_points)] 
            for i in range(n_points)
        ])
        
        shape_contexts = []
        for i in range(n_points):
            diff = sampled_points - sampled_points[i]
            distances = np.sqrt(np.sum(diff**2, axis=1))
            angles = np.arctan2(diff[:, 1], diff[:, 0])
            
            angles[angles < 0] += 2 * np.pi
            
            max_dist = distances.max()
            min_dist = distances[distances > 0].min()
            log_r_bins = np.logspace(np.log10(min_dist), np.log10(max_dist), n_bins_r)
            theta_bins = np.linspace(0, 2*np.pi, n_bins_theta+1)
            
            hist, _, _ = np.histogram2d(
                distances, angles, 
                bins=[log_r_bins, theta_bins]
            )
            shape_contexts.append(hist.flatten())
            
        return np.array(shape_contexts)

    def dtw_distance(self, seq1, seq2):
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.inf * np.ones((n + 1, m + 1))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        return dtw_matrix[n, m]

    def evaluate_contour_match(self, contour1, contour2):
        try:
            contour1 = contour1.squeeze()
            contour2 = contour2.squeeze()
            
            if contour1.ndim != 2 or contour2.ndim != 2:
                return 0.0
            
            if len(contour1) < 5 or len(contour2) < 5:
                return 0.0
            
            sc1 = self.compute_shape_context(contour1)
            sc2 = self.compute_shape_context(contour2)
            
            dtw_dist = self.dtw_distance(sc1, sc2)
            
            max_possible_dist = np.sqrt(sc1.shape[1])
            match_score = 1.0 - (dtw_dist / (max_possible_dist + 1e-5))
            
            return np.clip(match_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"轮廓匹配计算错误: {str(e)}")
            return 0.0