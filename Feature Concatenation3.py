import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time

class CylinderImageStitcher2D:
    def __init__(self, min_matches=10, ransac_thresh=5.0):
        """
        初始化二维拼接器
        
        Args:
            min_matches: 最小匹配点数
            ransac_thresh: RANSAC阈值
        """
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        
        # 初始化SIFT检测器
        self.sift = cv2.SIFT_create()
        
        # 使用FLANN匹配器
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def detect_and_compute(self, image):
        """提取SIFT特征"""
        if image is None:
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_images(self, desc1, desc2):
        """匹配特征点"""
        if desc1 is None or desc2 is None:
            return None
        
        try:
            # 使用knn匹配
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            
            # 应用Lowe's ratio test筛选好的匹配
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            return good_matches
        except Exception as e:
            print(f"匹配过程中出错: {e}")
            return None
    
    def find_homography(self, kp1, kp2, matches):
        """使用RANSAC计算单应性矩阵"""
        if matches is None or len(matches) < self.min_matches:
            return None
        
        try:
            # 提取匹配点
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # 使用RANSAC计算单应性矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                        ransacReprojThreshold=self.ransac_thresh)
            
            if H is None:
                return None
            
            # 统计内点数量
            inliers = mask.ravel().tolist().count(1)
            inlier_ratio = inliers / len(matches)
            
            return H, inlier_ratio, matches, mask
        except Exception as e:
            print(f"计算单应性矩阵时出错: {e}")
            return None
    
    def stitch_pair_2d(self, img1, img2, H):
        """二维拼接两张图像"""
        if H is None:
            print("单应性矩阵为空，无法拼接")
            return img1
        
        try:
            # 获取图像尺寸
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # 获取img1的四个角点
            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            
            # 将img1的角点变换到img2的坐标系
            corners2 = cv2.perspectiveTransform(corners1, H)
            
            # 计算拼接后图像的边界
            corners_all = np.concatenate((corners2.reshape(4, 2), 
                                         [[0, 0], [0, h2], [w2, h2], [w2, 0]]), axis=0)
            
            x_min, y_min = np.min(corners_all, axis=0).astype(int)
            x_max, y_max = np.max(corners_all, axis=0).astype(int)
            
            # 计算平移矩阵
            tx = -x_min if x_min < 0 else 0
            ty = -y_min if y_min < 0 else 0
            
            # 创建拼接画布
            result_width = int(x_max - x_min)
            result_height = int(y_max - y_min)
            
            # 组合变换矩阵（先透视变换再平移）
            T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            M = T @ H
            
            # 变换img1到新的画布
            warped_img1 = cv2.warpPerspective(img1, M, (result_width, result_height))
            
            # 将img2放置到画布上
            result = warped_img1.copy()
            
            # 计算img2在画布上的位置
            start_x = tx
            start_y = ty
            end_x = start_x + w2
            end_y = start_y + h2
            
            # 确保位置在图像范围内
            if start_x >= 0 and start_y >= 0 and end_x <= result_width and end_y <= result_height:
                result[start_y:end_y, start_x:end_x] = img2
            else:
                # 如果超出范围，调整位置
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(result_width, end_x)
                end_y = min(result_height, end_y)
                
                # 计算实际能放置的img2部分
                img2_start_x = max(0, -tx)
                img2_start_y = max(0, -ty)
                img2_end_x = min(w2, result_width - tx)
                img2_end_y = min(h2, result_height - ty)
                
                # 放置img2的有效部分
                if img2_end_x > img2_start_x and img2_end_y > img2_start_y:
                    result[start_y:end_y, start_x:end_x] = img2[img2_start_y:img2_end_y, img2_start_x:img2_end_x]
            
            # 创建重叠区域掩码并进行融合
            result = self.blend_overlap(result, warped_img1, img2, start_x, start_y, h2, w2)
            
            return result
        except Exception as e:
            print(f"二维拼接过程中出错: {e}")
            return img1
    
    def blend_overlap(self, result, warped_img1, img2, tx, ty, h2, w2):
        """融合重叠区域"""
        try:
            # 获取图像尺寸
            h1, w1 = result.shape[:2]
            
            # 创建重叠区域掩码
            overlap_mask = np.zeros((h1, w1), dtype=np.uint8)
            
            # img2在画布上的区域
            img2_area = np.zeros((h1, w1), dtype=np.uint8)
            
            # 计算img2的实际放置区域
            start_x = tx
            start_y = ty
            end_x = min(tx + w2, w1)
            end_y = min(ty + h2, h1)
            
            if start_x < end_x and start_y < end_y:
                img2_area[start_y:end_y, start_x:end_x] = 255
            
            # warped_img1的非零区域
            warped_gray = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
            warped_mask = (warped_gray > 10).astype(np.uint8) * 255
            
            # 找到重叠区域
            overlap = cv2.bitwise_and(warped_mask, img2_area)
            
            # 对重叠区域进行加权融合
            if np.any(overlap):
                # 计算距离变换作为权重
                dist1 = cv2.distanceTransform(255 - overlap, cv2.DIST_L2, 5)
                dist2 = cv2.distanceTransform(overlap, cv2.DIST_L2, 5)
                
                # 归一化权重
                weight1 = dist1 / (dist1 + dist2 + 1e-7)
                weight2 = dist2 / (dist1 + dist2 + 1e-7)
                
                # 扩展维度用于彩色图像
                weight1 = np.stack([weight1] * 3, axis=2)
                weight2 = np.stack([weight2] * 3, axis=2)
                
                # 提取重叠区域
                y, x = np.where(overlap)
                if len(y) > 0:
                    y_min, y_max = y.min(), y.max()
                    x_min, x_max = x.min(), x.max()
                    
                    # 确保ROI在图像范围内
                    y_min = max(0, y_min)
                    y_max = min(h1, y_max)
                    x_min = max(0, x_min)
                    x_max = min(w1, x_max)
                    
                    if y_max > y_min and x_max > x_min:
                        roi1 = warped_img1[y_min:y_max, x_min:x_max]
                        roi2 = result[y_min:y_max, x_min:x_max]
                        
                        # 加权融合
                        blended = (roi1 * weight1[y_min:y_max, x_min:x_max] + 
                                  roi2 * weight2[y_min:y_max, x_min:x_max])
                        
                        result[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)
            
            return result
        except Exception as e:
            print(f"融合重叠区域时出错: {e}")
            return result
    
    def stitch_horizontal(self, images):
        """水平拼接多张图像"""
        if len(images) < 2:
            return images[0] if images else None
        
        print(f"水平拼接 {len(images)} 张图像...")
        
        # 从第一张图像开始
        result = images[0].copy()
        
        for i in range(1, len(images)):
            print(f"  水平拼接第 {i+1} 张图像...")
            
            # 提取特征
            kp1, desc1 = self.detect_and_compute(result)
            kp2, desc2 = self.detect_and_compute(images[i])
            
            if desc1 is None or desc2 is None:
                print(f"  图像 {i+1} 无法提取特征，使用简单水平拼接")
                result = self.simple_horizontal_stitch(result, images[i])
                continue
            
            # 匹配特征
            matches = self.match_images(desc1, desc2)
            
            if matches is None or len(matches) < self.min_matches:
                print(f"  图像 {i+1} 匹配点不足，使用简单水平拼接")
                result = self.simple_horizontal_stitch(result, images[i])
                continue
            
            # 计算单应性矩阵
            homography_result = self.find_homography(kp1, kp2, matches)
            
            if homography_result is None:
                print(f"  无法计算单应性矩阵，使用简单水平拼接")
                result = self.simple_horizontal_stitch(result, images[i])
                continue
            
            H, inlier_ratio, _, _ = homography_result
            
            print(f"  匹配点: {len(matches)}, 内点率: {inlier_ratio:.2%}")
            
            # 检查变换矩阵是否主要是平移
            if self.is_mainly_translation(H):
                print("  检测到主要是平移变换，使用优化拼接")
                result = self.stitch_translation(result, images[i], H)
            else:
                # 使用一般的透视变换拼接
                new_result = self.stitch_pair_2d(result, images[i], H)
                if new_result is not None:
                    result = new_result
                else:
                    print(f"  图像 {i+1} 拼接失败，使用简单水平拼接")
                    result = self.simple_horizontal_stitch(result, images[i])
        
        return result
    
    def is_mainly_translation(self, H, threshold=0.1):
        """判断变换矩阵是否主要是平移"""
        if H is None:
            return False
        
        # 检查旋转和缩放分量
        rotation_scale = np.array([
            [H[0, 0] - 1, H[0, 1]],
            [H[1, 0], H[1, 1] - 1]
        ])
        
        # 计算矩阵的范数
        norm = np.linalg.norm(rotation_scale)
        
        return norm < threshold
    
    def stitch_translation(self, img1, img2, H):
        """专门处理平移变换的拼接"""
        try:
            # 获取图像尺寸
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # 提取平移分量
            dx = H[0, 2]
            dy = H[1, 2]
            
            # 创建平移矩阵（简化版，假设只有平移）
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            
            # 计算新图像的边界
            corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners2 = np.float32([[dx, dy], [w2 + dx, dy], 
                                  [w2 + dx, h2 + dy], [dx, h2 + dy]])
            
            all_corners = np.vstack([corners1, corners2])
            
            x_min = np.floor(np.min(all_corners[:, 0])).astype(int)
            x_max = np.ceil(np.max(all_corners[:, 0])).astype(int)
            y_min = np.floor(np.min(all_corners[:, 1])).astype(int)
            y_max = np.ceil(np.max(all_corners[:, 1])).astype(int)
            
            # 计算平移量，使所有像素坐标为正值
            tx = -x_min if x_min < 0 else 0
            ty = -y_min if y_min < 0 else 0
            
            # 调整平移矩阵
            M[0, 2] += tx
            M[1, 2] += ty
            
            # 创建新图像
            new_width = x_max - x_min
            new_height = y_max - y_min
            
            # 确保新尺寸为正
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            # 变换第一张图像
            warped = cv2.warpAffine(img1, M, (new_width, new_height))
            
            # 计算第二张图像的位置
            start_x = tx + int(round(dx))
            start_y = ty + int(round(dy))
            end_x = start_x + w2
            end_y = start_y + h2
            
            # 确保位置在图像范围内
            if start_x >= 0 and start_y >= 0 and end_x <= new_width and end_y <= new_height:
                # 直接放置，尺寸应该完全匹配
                warped[start_y:end_y, start_x:end_x] = img2
            else:
                # 如果超出范围，调整位置
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(new_width, end_x)
                end_y = min(new_height, end_y)
                
                # 计算实际能放置的img2部分
                img2_start_x = max(0, -dx - tx)
                img2_start_y = max(0, -dy - ty)
                img2_end_x = min(w2, new_width - start_x)
                img2_end_y = min(h2, new_height - start_y)
                
                # 放置img2的有效部分
                if img2_end_x > img2_start_x and img2_end_y > img2_start_y:
                    warped[start_y:end_y, start_x:end_x] = img2[img2_start_y:img2_end_y, img2_start_x:img2_end_x]
            
            return warped
        except Exception as e:
            print(f"平移拼接失败: {e}")
            return self.simple_horizontal_stitch(img1, img2)
    
    def stitch_vertical(self, images):
        """垂直拼接多张图像"""
        if len(images) < 2:
            return images[0] if images else None
        
        print(f"垂直拼接 {len(images)} 张图像...")
        
        # 从第一张图像开始
        result = images[0].copy()
        
        for i in range(1, len(images)):
            print(f"  垂直拼接第 {i+1} 张图像...")
            
            # 提取特征
            kp1, desc1 = self.detect_and_compute(result)
            kp2, desc2 = self.detect_and_compute(images[i])
            
            if desc1 is None or desc2 is None:
                print(f"  图像 {i+1} 无法提取特征，使用简单垂直拼接")
                result = self.simple_vertical_stitch(result, images[i])
                continue
            
            # 匹配特征
            matches = self.match_images(desc1, desc2)
            
            if matches is None or len(matches) < self.min_matches:
                print(f"  图像 {i+1} 匹配点不足，使用简单垂直拼接")
                result = self.simple_vertical_stitch(result, images[i])
                continue
            
            # 计算单应性矩阵
            homography_result = self.find_homography(kp1, kp2, matches)
            
            if homography_result is None:
                print(f"  无法计算单应性矩阵，使用简单垂直拼接")
                result = self.simple_vertical_stitch(result, images[i])
                continue
            
            H, inlier_ratio, _, _ = homography_result
            
            print(f"  匹配点: {len(matches)}, 内点率: {inlier_ratio:.2%}")
            
            # 拼接图像
            new_result = self.stitch_pair_2d(result, images[i], H)
            if new_result is not None:
                result = new_result
            else:
                print(f"  图像 {i+1} 拼接失败，使用简单垂直拼接")
                result = self.simple_vertical_stitch(result, images[i])
        
        return result
    
    def simple_horizontal_stitch(self, img1, img2, overlap_ratio=0.3):
        """简单水平拼接（特征匹配失败时使用）"""
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # 计算重叠区域宽度
            overlap_width = int(min(w1, w2) * overlap_ratio)
            
            # 创建新图像
            new_width = w1 + w2 - overlap_width
            new_height = max(h1, h2)
            
            result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            
            # 放置第一张图像
            result[:h1, :w1] = img1
            
            # 放置第二张图像（考虑重叠）
            start_x = w1 - overlap_width
            result[:h2, start_x:start_x+w2] = img2
            
            # 融合重叠区域
            if overlap_width > 0:
                blend_start = start_x
                blend_end = w1
                
                for x in range(blend_start, blend_end):
                    alpha = (x - blend_start) / (blend_end - blend_start)  # 线性渐变
                    # 确保索引有效
                    if x < new_width and (x - start_x) < w2:
                        result[:min(h1, h2), x] = (1 - alpha) * img1[:min(h1, h2), x] + alpha * img2[:min(h1, h2), x - start_x]
            
            return result
        except Exception as e:
            print(f"简单水平拼接失败: {e}")
            # 如果失败，返回并排的图像
            return self.side_by_side_stitch(img1, img2)
    
    def simple_vertical_stitch(self, img1, img2, overlap_ratio=0.3):
        """简单垂直拼接（特征匹配失败时使用）"""
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # 计算重叠区域高度
            overlap_height = int(min(h1, h2) * overlap_ratio)
            
            # 创建新图像
            new_width = max(w1, w2)
            new_height = h1 + h2 - overlap_height
            
            result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            
            # 放置第一张图像
            result[:h1, :w1] = img1
            
            # 放置第二张图像（考虑重叠）
            start_y = h1 - overlap_height
            result[start_y:start_y+h2, :w2] = img2
            
            # 融合重叠区域
            if overlap_height > 0:
                blend_start = start_y
                blend_end = h1
                
                for y in range(blend_start, blend_end):
                    alpha = (y - blend_start) / (blend_end - blend_start)  # 线性渐变
                    # 确保索引有效
                    if y < new_height and (y - start_y) < h2:
                        result[y, :min(w1, w2)] = (1 - alpha) * img1[y, :min(w1, w2)] + alpha * img2[y - start_y, :min(w1, w2)]
            
            return result
        except Exception as e:
            print(f"简单垂直拼接失败: {e}")
            # 如果失败，返回上下排列的图像
            return self.vertical_stack_stitch(img1, img2)
    
    def side_by_side_stitch(self, img1, img2):
        """简单的并排拼接（无重叠）"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        new_height = max(h1, h2)
        new_width = w1 + w2
        
        result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        result[:h1, :w1] = img1
        result[:h2, w1:w1+w2] = img2
        
        return result
    
    def vertical_stack_stitch(self, img1, img2):
        """简单的垂直堆叠拼接（无重叠）"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        new_width = max(w1, w2)
        new_height = h1 + h2
        
        result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        result[:h1, :w1] = img1
        result[h1:h1+h2, :w2] = img2
        
        return result
    
    def auto_stitch_2d(self, images):
        """自动二维拼接（根据图像特征自动判断布局）"""
        if len(images) < 2:
            return images[0] if images else None
        
        print("开始自动二维拼接...")
        
        # 分析图像布局
        layout = self.analyze_image_layout(images)
        print(f"检测到的图像布局: {layout}")
        
        if layout == "vertical":
            print("使用垂直拼接策略...")
            return self.stitch_vertical(images)
        else:
            print("使用水平拼接策略...")
            # 对于管道内壁展开图，可能是网格布局
            # 尝试自动检测是否可能是多行多列
            if len(images) > 4:
                print("图像数量较多，尝试网格拼接...")
                return self.stitch_grid(images)
            else:
                return self.stitch_horizontal(images)
    
    def analyze_image_layout(self, images):
        """分析图像布局（水平或垂直排列）"""
        if len(images) < 2:
            return "horizontal"  # 默认
        
        # 获取第一张图像的尺寸
        h_ref, w_ref = images[0].shape[:2]
        
        # 分析后续图像的相对位置
        horizontal_scores = []
        vertical_scores = []
        
        for i in range(1, len(images)):
            img = images[i]
            h, w = img.shape[:2]
            
            # 计算宽高比
            aspect_ratio_ref = w_ref / h_ref
            aspect_ratio = w / h
            
            # 如果宽高比相似，可能是水平排列
            if abs(aspect_ratio - aspect_ratio_ref) / aspect_ratio_ref < 0.3:
                horizontal_scores.append(1)
            else:
                horizontal_scores.append(0)
            
            # 如果宽度相近但高度不同，可能是垂直排列
            if abs(w - w_ref) / w_ref < 0.2 and abs(h - h_ref) / h_ref > 0.3:
                vertical_scores.append(1)
            else:
                vertical_scores.append(0)
        
        # 判断布局
        avg_horizontal = np.mean(horizontal_scores) if horizontal_scores else 0
        avg_vertical = np.mean(vertical_scores) if vertical_scores else 0
        
        print(f"水平布局得分: {avg_horizontal:.2f}, 垂直布局得分: {avg_vertical:.2f}")
        
        if avg_vertical > avg_horizontal:
            return "vertical"
        else:
            return "horizontal"
    
    def stitch_grid(self, images, rows=None, cols=None):
        """
        网格拼接（支持多行多列）
        
        Args:
            images: 图像列表
            rows: 行数（自动计算如果为None）
            cols: 列数（自动计算如果为None）
        """
        if len(images) < 2:
            return images[0] if images else None
        
        print(f"开始网格拼接，共 {len(images)} 张图像")
        
        # 自动确定行列数
        if rows is None or cols is None:
            if rows is None and cols is None:
                # 尝试找到最接近平方根的值
                n = len(images)
                cols = int(np.ceil(np.sqrt(n)))
                rows = int(np.ceil(n / cols))
            elif rows is None:
                rows = int(np.ceil(len(images) / cols))
            elif cols is None:
                cols = int(np.ceil(len(images) / rows))
        
        print(f"网格布局: {rows}行 × {cols}列")
        
        # 按行拼接
        row_results = []
        for r in range(rows):
            start_idx = r * cols
            end_idx = min(start_idx + cols, len(images))
            row_images = images[start_idx:end_idx]
            
            if len(row_images) > 1:
                print(f"拼接第 {r+1} 行 ({len(row_images)} 张图像)...")
                row_result = self.stitch_horizontal(row_images)
                if row_result is not None:
                    row_results.append(row_result)
                else:
                    print(f"第 {r+1} 行拼接失败，使用第一张图像")
                    row_results.append(row_images[0] if row_images else None)
            elif len(row_images) == 1:
                row_results.append(row_images[0])
            else:
                break
        
        # 过滤掉None值
        row_results = [r for r in row_results if r is not None]
        
        # 按列拼接（垂直拼接各行）
        if len(row_results) > 1:
            print(f"垂直拼接 {len(row_results)} 行...")
            final_result = self.stitch_vertical(row_results)
        elif len(row_results) == 1:
            final_result = row_results[0]
        else:
            final_result = None
        
        return final_result

def load_images(image_folder: str, patterns: List[str] = None) -> List[np.ndarray]:
    """加载图像，支持多种格式"""
    if patterns is None:
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    
    image_paths = []
    for pattern in patterns:
        image_paths.extend(sorted(Path(image_folder).glob(pattern)))
        image_paths.extend(sorted(Path(image_folder).glob(pattern.upper())))
    
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
            print(f"加载: {path.name}, 尺寸: {img.shape}")
        else:
            print(f"无法加载: {path.name}")
    
    return images

def preprocess_images(images):
    """预处理图像以提高拼接效果"""
    processed = []
    for i, img in enumerate(images):
        try:
            # 直方图均衡化（LAB颜色空间）
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 轻度去噪
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            processed.append(denoised)
        except Exception as e:
            print(f"预处理图像 {i+1} 时出错: {e}")
            processed.append(img)  # 使用原始图像
    
    return processed

def show_image_with_opencv(image, title="图像显示", max_display_size=1200):
    """使用OpenCV显示单个图像"""
    if image is None:
        print(f"无法显示: {title} (图像为None)")
        return
    
    # 调整图像大小以便显示
    h, w = image.shape[:2]
    if w > max_display_size or h > max_display_size:
        scale = min(max_display_size / w, max_display_size / h)
        new_width = int(w * scale)
        new_height = int(h * scale)
        resized = cv2.resize(image, (new_width, new_height))
    else:
        resized = image
    
    cv2.imshow(title, resized)
    print(f"显示: {title}, 尺寸: {image.shape} -> {resized.shape}")
    
def save_and_display_results(results, titles):
    """保存并显示拼接结果"""
    if not results:
        print("没有结果可显示")
        return
    
    print("\n保存拼接结果...")
    for i, (result, title) in enumerate(zip(results, titles)):
        if result is not None:
            filename = f"stitched_{title}.jpg"
            cv2.imwrite(filename, result)
            print(f"  {title}结果已保存为: {filename}, 尺寸: {result.shape}")
            
            # 显示图像
            show_image_with_opencv(result, title)
        else:
            print(f"  {title}失败，无结果保存")
    
    print("\n按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 配置参数
    IMAGE_FOLDER = "ppp"  # 修改为你的图像文件夹路径
    
    print("=" * 60)
    print("管道内壁展开图二维拼接系统")
    print("=" * 60)
    
    # 加载图像
    print("\n[1/5] 加载图像...")
    images = load_images(IMAGE_FOLDER)
    
    if len(images) < 2:
        print("需要至少2张图像进行拼接")
        return
    
    print(f"成功加载 {len(images)} 张图像")
    
    # 显示原始图像
    print("\n[2/5] 显示原始图像...")
    for i, img in enumerate(images):
        show_image_with_opencv(img, f"原始图像 {i+1}")
    
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 预处理图像
    print("\n[3/5] 预处理图像...")
    processed_images = preprocess_images(images)
    
    # 创建拼接器
    stitcher = CylinderImageStitcher2D(min_matches=10, ransac_thresh=3.0)
    
    # 用户选择拼接模式
    print("\n[4/5] 选择拼接模式:")
    print("  1. 自动模式（推荐）")
    print("  2. 水平拼接")
    print("  3. 垂直拼接")
    print("  4. 网格拼接")
    
    choice = 2 #input("请输入选择 (1-4, 默认1): ").strip()
    
    results = []
    titles = []
    
    # 执行拼接
    print("\n[5/5] 执行拼接...")
    
    if choice == "2":
        print("执行水平拼接...")
        result = stitcher.stitch_horizontal(processed_images)
        results.append(result)
        titles.append("水平拼接")
        
    elif choice == "3":
        print("执行垂直拼接...")
        result = stitcher.stitch_vertical(processed_images)
        results.append(result)
        titles.append("垂直拼接")
        
    elif choice == "4":
        # 询问网格布局
        if len(images) > 1:
            print(f"当前有 {len(images)} 张图像")
            rows = input(f"请输入行数 (建议值: {int(np.ceil(np.sqrt(len(images))))}): ").strip()
            cols = input(f"请输入列数 (建议值: {int(np.ceil(len(images) / int(rows) if rows else np.ceil(np.sqrt(len(images)))))}): ").strip()
            
            try:
                rows = int(rows) if rows else None
                cols = int(cols) if cols else None
            except:
                rows = None
                cols = None
            
            print("执行网格拼接...")
            result = stitcher.stitch_grid(processed_images, rows, cols)
            results.append(result)
            titles.append("网格拼接")
        
    else:  # 默认自动模式
        print("执行自动拼接...")
        result = stitcher.auto_stitch_2d(processed_images)
        results.append(result)
        titles.append("自动拼接")
    
    # 保存并显示结果
    save_and_display_results(results, titles)
    
    print("\n" + "=" * 60)
    print("拼接完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()