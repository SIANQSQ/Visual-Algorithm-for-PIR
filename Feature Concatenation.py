import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import time

class CylinderImageStitcher:
    def __init__(self, min_matches=10, ransac_thresh=5.0):
        """
        初始化拼接器
        
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
        
        # 提取匹配点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            # 使用RANSAC计算单应性矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                        ransacReprojThreshold=self.ransac_thresh)
            
            # 统计内点数量
            inliers = mask.ravel().tolist().count(1)
            inlier_ratio = inliers / len(matches)
            
            return H, inlier_ratio, matches, mask
        except Exception as e:
            print(f"计算单应性矩阵时出错: {e}")
            return None
    
    def stitch_pair(self, img1, img2, H):
        """拼接两张图像"""
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
            all_corners = np.concatenate((corners2.reshape(4, 2), 
                                         [[0, 0], [0, h2], [w2, h2], [w2, 0]]), axis=0)
            
            x_min, y_min = np.min(all_corners, axis=0).astype(int)
            x_max, y_max = np.max(all_corners, axis=0).astype(int)
            
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
            result[ty:ty+h2, tx:tx+w2] = img2
            
            # 创建重叠区域掩码
            overlap_mask = np.zeros((result_height, result_width), dtype=np.uint8)
            warped_mask = cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8) * 255, 
                                             M, (result_width, result_height))
            overlap_mask[ty:ty+h2, tx:tx+w2] += 255
            
            # 找到重叠区域
            overlap = cv2.bitwise_and(warped_mask, overlap_mask)
            
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
                    y_min_roi, y_max_roi = y.min(), y.max()
                    x_min_roi, x_max_roi = x.min(), x.max()
                    
                    roi1 = warped_img1[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
                    roi2 = result[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
                    
                    # 加权融合
                    blended = (roi1 * weight1[y_min_roi:y_max_roi, x_min_roi:x_max_roi] + 
                              roi2 * weight2[y_min_roi:y_max_roi, x_min_roi:x_max_roi])
                    
                    result[y_min_roi:y_max_roi, x_min_roi:x_max_roi] = blended.astype(np.uint8)
            
            return result
        except Exception as e:
            print(f"拼接过程中出错: {e}")
            return img1
    
    def stitch_all_pairs_fixed(self, images: List[np.ndarray]):
        """修复后的全局拼接方法"""
        if len(images) < 2:
            return images[0] if images else None
        
        print(f"开始全局拼接，共 {len(images)} 张图像")
        
        # 提取所有图像的特征
        print("1. 提取所有图像的特征...")
        features = []
        for i, img in enumerate(images):
            kp, desc = self.detect_and_compute(img)
            if desc is not None and len(kp) >= self.min_matches:
                features.append((kp, desc, img))
                print(f"   图像 {i+1}: {len(kp)} 个特征点")
            else:
                print(f"   图像 {i+1}: 特征不足或无法提取特征")
        
        if len(features) < 2:
            print("有效图像不足2张，无法进行全局拼接")
            return None
        
        # 使用简单的增量拼接策略
        print("2. 使用增量拼接策略...")
        
        # 选择特征点最多的图像作为起始点
        start_idx = max(range(len(features)), key=lambda i: len(features[i][0]))
        print(f"   选择图像 {start_idx+1} 作为起始图像")
        
        # 初始化
        base_img = features[start_idx][2].copy()
        base_kp, base_desc, _ = features[start_idx]
        used = [False] * len(features)
        used[start_idx] = True
        
        # 创建待拼接列表
        to_stitch = [i for i in range(len(features)) if not used[i]]
        
        iteration = 0
        while to_stitch and iteration < len(features) * 2:
            iteration += 1
            best_idx = -1
            best_score = -1
            best_H = None
            
            print(f"   第 {iteration} 轮: 待拼接图像 {len(to_stitch)} 张")
            
            # 寻找与当前结果最佳匹配的图像
            for idx in to_stitch:
                kp2, desc2, _ = features[idx]
                
                # 匹配特征
                matches = self.match_images(base_desc, desc2)
                if matches is None or len(matches) < self.min_matches:
                    continue
                
                # 计算单应性矩阵
                homography_result = self.find_homography(base_kp, kp2, matches)
                if homography_result is None:
                    continue
                
                H, inlier_ratio, _, _ = homography_result
                
                if inlier_ratio > best_score:
                    best_score = inlier_ratio
                    best_idx = idx
                    best_H = H
            
            if best_idx == -1:
                print("   未找到可匹配的图像，停止拼接")
                break
            
            print(f"   找到最佳匹配: 图像 {best_idx+1}, 匹配度: {best_score:.2%}")
            
            # 拼接图像
            target_img = features[best_idx][2]
            base_img = self.stitch_pair(base_img, target_img, best_H)
            
            if base_img is None:
                print("   拼接失败，跳过该图像")
                used[best_idx] = True
                to_stitch.remove(best_idx)
                continue
            
            # 更新基础特征
            base_kp, base_desc = self.detect_and_compute(base_img)
            if base_desc is None:
                print("   新拼接图像无法提取特征，停止拼接")
                break
            
            # 标记为已使用
            used[best_idx] = True
            to_stitch.remove(best_idx)
            
            print(f"   成功拼接图像 {best_idx+1}，当前结果尺寸: {base_img.shape}")
        
        # 检查是否有未使用的图像
        unused = sum(1 for u in used if not u)
        if unused > 0:
            print(f"警告: {unused} 张图像未能拼接")
        
        return base_img
    
    def stitch_sequential_fixed(self, images: List[np.ndarray]):
        """修复后的顺序拼接方法"""
        if len(images) < 2:
            return images[0] if images else None
        
        print("开始顺序拼接...")
        
        # 从第一张图像开始
        result = images[0].copy()
        
        for i in range(1, len(images)):
            print(f"  拼接第 {i+1} 张图像...")
            
            # 提取特征
            kp1, desc1 = self.detect_and_compute(result)
            kp2, desc2 = self.detect_and_compute(images[i])
            
            if desc1 is None or desc2 is None:
                print(f"  图像 {i+1} 无法提取特征，跳过")
                continue
            
            # 匹配特征
            matches = self.match_images(desc1, desc2)
            
            if matches is None or len(matches) < self.min_matches:
                print(f"  图像 {i+1} 匹配点不足 ({len(matches) if matches else 0} < {self.min_matches})，尝试简单拼接")
                # 尝试简单水平拼接
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue
            
            # 计算单应性矩阵
            homography_result = self.find_homography(kp1, kp2, matches)
            
            if homography_result is None:
                print(f"  无法计算图像 {i+1} 的单应性矩阵，尝试简单拼接")
                # 尝试简单水平拼接
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue
            
            H, inlier_ratio, _, _ = homography_result
            
            print(f"  匹配点: {len(matches)}, 内点率: {inlier_ratio:.2%}")
            
            # 拼接图像
            new_result = self.stitch_pair(result, images[i], H)
            if new_result is not None:
                result = new_result
            else:
                print(f"  图像 {i+1} 拼接失败，保留当前结果")
        
        return result
    
    def stitch_horizontal(self, images: List[np.ndarray]):
        """水平拼接方法（针对管道展开图的特性）"""
        if len(images) < 2:
            return images[0] if images else None
        
        print("开始水平拼接（针对管道展开图）...")
        
        # 首先确保所有图像高度相同
        heights = [img.shape[0] for img in images]
        min_height = min(heights)
        max_height = max(heights)
        
        print(f"  图像高度范围: {min_height} - {max_height}")
        
        # 如果高度差异太大，可能需要调整
        if max_height > min_height * 1.2:
            print("  图像高度差异较大，进行高度对齐...")
            # 调整所有图像到最小高度
            aligned_images = []
            for img in images:
                if img.shape[0] > min_height:
                    # 计算缩放比例
                    scale = min_height / img.shape[0]
                    new_width = int(img.shape[1] * scale)
                    resized = cv2.resize(img, (new_width, min_height))
                    aligned_images.append(resized)
                else:
                    aligned_images.append(img)
            images = aligned_images
        
        # 尝试使用特征匹配进行水平拼接
        result = images[0].copy()
        
        for i in range(1, len(images)):
            print(f"  拼接第 {i+1} 张图像...")
            
            # 提取特征
            kp1, desc1 = self.detect_and_compute(result)
            kp2, desc2 = self.detect_and_compute(images[i])
            
            if desc1 is None or desc2 is None:
                print(f"  图像 {i+1} 无法提取特征，使用简单水平拼接")
                # 简单水平拼接
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue
            
            # 匹配特征
            matches = self.match_images(desc1, desc2)
            
            if matches is None or len(matches) < self.min_matches:
                print(f"  图像 {i+1} 匹配点不足，使用简单水平拼接")
                # 简单水平拼接
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue
            
            # 计算单应性矩阵
            homography_result = self.find_homography(kp1, kp2, matches)
            
            if homography_result is None:
                print(f"  无法计算单应性矩阵，使用简单水平拼接")
                # 简单水平拼接
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue
            
            H, inlier_ratio, _, _ = homography_result
            
            print(f"  匹配点: {len(matches)}, 内点率: {inlier_ratio:.2%}")
            
            # 对于水平拼接，我们期望主要是水平方向的变换
            # 可以检查变换矩阵，如果是近似单位矩阵加平移，可以直接使用平移
            if abs(H[0, 0] - 1.0) < 0.1 and abs(H[1, 1] - 1.0) < 0.1 and abs(H[0, 1]) < 0.1 and abs(H[1, 0]) < 0.1:
                print("  检测到主要是平移变换，使用优化拼接")
                # 提取平移分量
                dx = H[0, 2]
                dy = H[1, 2]
                
                # 创建平移矩阵
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                
                # 计算拼接后的大小
                h1, w1 = result.shape[:2]
                h2, w2 = images[i].shape[:2]
                
                # 计算新图像的边界
                corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
                corners2 = np.float32([[dx, dy], [w2 + dx, dy], 
                                      [w2 + dx, h2 + dy], [dx, h2 + dy]])
                
                all_corners = np.vstack([corners1, corners2])
                
                x_min = int(min(all_corners[:, 0]))
                x_max = int(max(all_corners[:, 0]))
                y_min = int(min(all_corners[:, 1]))
                y_max = int(max(all_corners[:, 1]))
                
                # 计算平移量，使所有像素坐标为正值
                tx = -x_min if x_min < 0 else 0
                ty = -y_min if y_min < 0 else 0
                
                # 调整平移矩阵
                M[0, 2] += tx
                M[1, 2] += ty
                
                # 创建新图像
                new_width = x_max - x_min
                new_height = y_max - y_min
                warped = cv2.warpAffine(result, M, (new_width, new_height))
                
                # 放置第二张图像
                warped[ty:ty+h2, tx + int(dx):tx + int(dx)+w2] = images[i]
                result = warped
            else:
                # 使用一般的透视变换拼接
                new_result = self.stitch_pair(result, images[i], H)
                if new_result is not None:
                    result = new_result
                else:
                    print(f"  图像 {i+1} 拼接失败，使用简单水平拼接")
                    # 简单水平拼接
                    h = max(result.shape[0], images[i].shape[0])
                    w = result.shape[1] + images[i].shape[1]
                    new_img = np.zeros((h, w, 3), dtype=np.uint8)
                    new_img[:result.shape[0], :result.shape[1]] = result
                    new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                    result = new_img
        
        return result

def load_images(image_folder: str, pattern: str = "*.jpg") -> List[np.ndarray]:
    """加载图像"""
    image_paths = sorted(Path(image_folder).glob(pattern))
    images = []
    
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
            print(f"加载: {path.name}, 尺寸: {img.shape}")
        else:
            print(f"无法加载: {path.name}")
    
    return images

def resize_image(image, max_width=1200):
    """按比例调整图像大小（用于显示）"""
    if image is None:
        return None
    
    if image.shape[1] > max_width:
        ratio = max_width / image.shape[1]
        new_width = max_width
        new_height = int(image.shape[0] * ratio)
        return cv2.resize(image, (new_width, new_height))
    return image

def show_images_with_opencv(images, titles, window_name="result", max_width=1200):
    """使用OpenCV显示图像"""
    if not images:
        print("没有图像可显示")
        return
    
    # 显示各个单独图像
    for i, (img, title) in enumerate(zip(images, titles)):
        if img is not None:
            resized = resize_image(img, max_width)
            cv2.imshow(f"{title}", resized)
    
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_duplicate_overlap(stitched_image, threshold=0.95):
    """
    检测并去除重复区域（基于图像自相似性）
    """
    if stitched_image is None:
        return None
    
    try:
        gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        
        # 使用模板匹配检测重复
        h, w = gray.shape
        template_size = min(100, w//10)  # 模板大小
        
        if template_size < 20:  # 模板太小，不进行检测
            return stitched_image
        
        # 提取左侧模板
        template = gray[:, :template_size]
        
        # 在整个图像中搜索相似区域
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        
        # 找到高度匹配的位置
        loc = np.where(res >= threshold)
        
        if len(loc[0]) > 0 and len(loc[1]) > 0:
            # 找到第一个匹配位置（最左侧）
            first_match = loc[1].min()
            
            if first_match > template_size and first_match < w - template_size:
                # 裁剪图像，去除左侧重复部分
                cropped = stitched_image[:, first_match:]
                print(f"检测到重复区域，裁剪位置: {first_match}")
                return cropped
        
        return stitched_image
    except Exception as e:
        print(f"去除重复区域时出错: {e}")
        return stitched_image

def preprocess_images(images):
    """预处理图像以提高拼接效果"""
    processed = []
    for img in images:
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
    
    return processed

def main():
    # 配置参数
    IMAGE_FOLDER = "fp"  # 修改为你的图像文件夹路径
    
    # 加载图像
    print("=" * 50)
    print("加载图像...")
    images = load_images(IMAGE_FOLDER)
    
    if len(images) < 2:
        print("需要至少2张图像进行拼接")
        return
    
    print(f"成功加载 {len(images)} 张图像")
    
    # 显示原始图像
    print("\n" + "=" * 50)
    print("显示原始图像...")
    #show_images_with_opencv(images, [f"原始图像 {i+1}" for i in range(len(images))], "原始图像")
    
    # 预处理图像
    print("\n" + "=" * 50)
    print("预处理图像...")
    processed_images = preprocess_images(images)
    
    # 创建拼接器（调整参数以获得更好的效果）
    stitcher = CylinderImageStitcher(min_matches=15, ransac_thresh=3.0)
    
    # 方法1: 水平拼接（针对管道展开图特性）
    print("\n" + "=" * 50)
    print("方法1: 水平拼接（针对管道展开图）")
    result_horizontal = stitcher.stitch_horizontal(processed_images)
    
    # 方法2: 顺序拼接
    print("\n" + "=" * 50)
    print("方法2: 顺序拼接")
    result_sequential = stitcher.stitch_sequential_fixed(processed_images)
    
    # 方法3: 全局拼接
    print("\n" + "=" * 50)
    print("方法3: 全局拼接")
    result_global = stitcher.stitch_all_pairs_fixed(processed_images)
    
    # 去除可能的重复区域
    print("\n" + "=" * 50)
    print("去除重复区域...")
    
    results = []
    titles = []
    
    if result_horizontal is not None:
        result_horizontal_clean = remove_duplicate_overlap(result_horizontal)
        results.append(result_horizontal_clean)
        titles.append("Horizontal")
        cv2.imwrite("stitched_horizontal.jpg", result_horizontal_clean)
        print(f"水平拼接结果已保存为: stitched_horizontal.jpg, 尺寸: {result_horizontal_clean.shape}")
    
    if result_sequential is not None:
        result_sequential_clean = remove_duplicate_overlap(result_sequential)
        results.append(result_sequential_clean)
        titles.append("Sequential")
        #cv2.imwrite("stitched_sequential.jpg", result_sequential_clean)
        print(f"顺序拼接结果已保存为: stitched_sequential.jpg, 尺寸: {result_sequential_clean.shape}")
    
    if result_global is not None:
        result_global_clean = remove_duplicate_overlap(result_global)
        results.append(result_global_clean)
        titles.append("Global")
        cv2.imwrite("stitched_global.jpg", result_global_clean)
        print(f"全局拼接结果已保存为: stitched_global.jpg, 尺寸: {result_global_clean.shape}")
    
    # 显示拼接结果
    print("\n" + "=" * 50)
    print("显示拼接结果...")
    if results:
        show_images_with_opencv(results, titles, "拼接结果对比")
    else:
        print("所有拼接方法都失败了")
    
    print("\n" + "=" * 50)
    print("拼接完成！")

if __name__ == "__main__":
    main()