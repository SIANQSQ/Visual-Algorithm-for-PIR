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

        self.sift = cv2.SIFT_create()

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
            matches = self.flann.knnMatch(desc1, desc2, k=2)

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

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                        ransacReprojThreshold=self.ransac_thresh)

            inliers = mask.ravel().tolist().count(1)
            inlier_ratio = inliers / len(matches)

            return H, inlier_ratio, matches, mask
        except Exception as e:
            print(f"计算单应性矩阵时出错: {e}")
            return None

    def blend_images(self, img1, img2, mask1, mask2):
        """改进的图像融合方法"""
        overlap = cv2.bitwise_and(mask1, mask2)

        if not np.any(overlap):
            result = np.where(mask1[:,:,np.newaxis] > 0, img1, img2)
            return result.astype(np.uint8)

        dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)

        weight1 = dist1 / (dist1 + dist2 + 1e-10)
        weight2 = 1.0 - weight1

        weight1 = np.stack([weight1] * 3, axis=2)
        weight2 = np.stack([weight2] * 3, axis=2)

        result = (img1 * weight1 + img2 * weight2).astype(np.uint8)

        result = np.where(mask1[:,:,np.newaxis] == 0, img2, result)
        result = np.where(mask2[:,:,np.newaxis] == 0, img1, result)

        return result

    def stitch_pair(self, img1, img2, H):
        """拼接两张图像"""
        if H is None:
            print("单应性矩阵为空，无法拼接")
            return img1

        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners2 = cv2.perspectiveTransform(corners1, H)

            all_corners = np.concatenate((corners2.reshape(4, 2),
                                         [[0, 0], [0, h2], [w2, h2], [w2, 0]]), axis=0)

            x_min, y_min = np.int32(np.floor(all_corners.min(axis=0)))
            x_max, y_max = np.int32(np.ceil(all_corners.max(axis=0)))

            tx = -x_min if x_min < 0 else 0
            ty = -y_min if y_min < 0 else 0

            result_width = int(x_max - x_min)
            result_height = int(y_max - y_min)

            T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
            M = T @ H

            warped_img1 = cv2.warpPerspective(img1, M, (result_width, result_height))
            warped_mask1 = cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8) * 255,
                                              M, (result_width, result_height))

            canvas_img2 = np.zeros((result_height, result_width, 3), dtype=np.uint8)
            canvas_mask2 = np.zeros((result_height, result_width), dtype=np.uint8)

            y_end = min(ty + h2, result_height)
            x_end = min(tx + w2, result_width)

            h2_crop = y_end - ty
            w2_crop = x_end - tx

            canvas_img2[ty:y_end, tx:x_end] = img2[:h2_crop, :w2_crop]
            canvas_mask2[ty:y_end, tx:x_end] = 255

            result = self.blend_images(warped_img1, canvas_img2, warped_mask1, canvas_mask2)

            return result
        except Exception as e:
            print(f"拼接过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return img1

    def stitch_all_pairs_fixed(self, images: List[np.ndarray]):
        """改进的全局拼接方法"""
        if len(images) < 2:
            return images[0] if images else None

        print(f"开始全局拼接，共 {len(images)} 张图像")

        print("1. 提取所有图像的特征...")
        features = []
        for i, img in enumerate(images):
            kp, desc = self.detect_and_compute(img)
            if desc is not None and len(kp) >= self.min_matches:
                features.append((kp, desc, img, i))
                print(f"   图像 {i+1}: {len(kp)} 个特征点")
            else:
                print(f"   图像 {i+1}: 特征不足或无法提取特征")

        if len(features) < 2:
            print("有效图像不足2张，无法进行全局拼接")
            return None

        print("2. 使用改进的增量拼接策略...")

        start_idx = max(range(len(features)), key=lambda i: len(features[i][0]))
        print(f"   选择图像 {features[start_idx][3]+1} 作为起始图像")

        base_img = features[start_idx][2].copy()
        base_kp, base_desc, _, _ = features[start_idx]
        used = [False] * len(features)
        used[start_idx] = True

        to_stitch = [i for i in range(len(features)) if not used[i]]

        iteration = 0
        max_iterations = len(features) * 3
        while to_stitch and iteration < max_iterations:
            iteration += 1
            best_idx = -1
            best_score = -1
            best_H = None
            best_matches_count = 0

            print(f"   第 {iteration} 轮: 待拼接图像 {len(to_stitch)} 张")

            for idx in to_stitch:
                kp2, desc2, _, orig_idx = features[idx]

                matches = self.match_images(base_desc, desc2)
                if matches is None or len(matches) < self.min_matches:
                    continue

                homography_result = self.find_homography(base_kp, kp2, matches)
                if homography_result is None:
                    continue

                H, inlier_ratio, _, _ = homography_result

                score = inlier_ratio * len(matches)

                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_H = H
                    best_matches_count = len(matches)

            if best_idx == -1:
                print("   未找到可匹配的图像，停止拼接")
                break

            orig_img_num = features[best_idx][3] + 1
            print(f"   找到最佳匹配: 图像 {orig_img_num}, 匹配点: {best_matches_count}, 内点率: {best_score/best_matches_count:.2%}")

            target_img = features[best_idx][2]
            new_base_img = self.stitch_pair(base_img, target_img, best_H)

            if new_base_img is None:
                print(f"   图像 {orig_img_num} 拼接失败，跳过")
                used[best_idx] = True
                to_stitch.remove(best_idx)
                continue

            base_img = new_base_img
            base_kp, base_desc = self.detect_and_compute(base_img)

            if base_desc is None or len(base_kp) < self.min_matches:
                print(f"   警告: 拼接后的图像特征不足，但继续拼接")

            used[best_idx] = True
            to_stitch.remove(best_idx)

            print(f"   成功拼接图像 {orig_img_num}，当前结果尺寸: {base_img.shape}")

        unused_count = sum(1 for u in used if not u)
        if unused_count > 0:
            print(f"警告: {unused_count} 张图像未能拼接")
        else:
            print("所有图像均已成功拼接！")

        return base_img

    def stitch_sequential_fixed(self, images: List[np.ndarray]):
        """修复后的顺序拼接方法"""
        if len(images) < 2:
            return images[0] if images else None

        print("开始顺序拼接...")

        result = images[0].copy()

        for i in range(1, len(images)):
            print(f"  拼接第 {i+1} 张图像...")

            kp1, desc1 = self.detect_and_compute(result)
            kp2, desc2 = self.detect_and_compute(images[i])

            if desc1 is None or desc2 is None:
                print(f"  图像 {i+1} 无法提取特征，跳过")
                continue

            matches = self.match_images(desc1, desc2)

            if matches is None or len(matches) < self.min_matches:
                print(f"  图像 {i+1} 匹配点不足 ({len(matches) if matches else 0} < {self.min_matches})，尝试简单拼接")
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue

            homography_result = self.find_homography(kp1, kp2, matches)

            if homography_result is None:
                print(f"  无法计算图像 {i+1} 的单应性矩阵，尝试简单拼接")
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue

            H, inlier_ratio, _, _ = homography_result

            print(f"  匹配点: {len(matches)}, 内点率: {inlier_ratio:.2%}")

            new_result = self.stitch_pair(result, images[i], H)
            if new_result is not None:
                result = new_result
            else:
                print(f"  图像 {i+1} 拼接失败，保留当前结果")

        return result

    def stitch_horizontal(self, images: List[np.ndarray]):
        """水平拼接方法（针对管道展开图的特性）- 修复版"""
        if len(images) < 2:
            return images[0] if images else None

        print("开始水平拼接（针对管道展开图）...")

        heights = [img.shape[0] for img in images]
        min_height = min(heights)
        max_height = max(heights)

        print(f"  图像高度范围: {min_height} - {max_height}")

        if max_height > min_height * 1.2:
            print("  图像高度差异较大，进行高度对齐...")
            aligned_images = []
            for img in images:
                if img.shape[0] > min_height:
                    scale = min_height / img.shape[0]
                    new_width = int(img.shape[1] * scale)
                    resized = cv2.resize(img, (new_width, min_height))
                    aligned_images.append(resized)
                else:
                    aligned_images.append(img)
            images = aligned_images

        result = images[0].copy()

        for i in range(1, len(images)):
            print(f"  拼接第 {i+1} 张图像...")

            kp1, desc1 = self.detect_and_compute(result)
            kp2, desc2 = self.detect_and_compute(images[i])

            if desc1 is None or desc2 is None:
                print(f"  图像 {i+1} 无法提取特征，使用简单水平拼接")
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue

            matches = self.match_images(desc1, desc2)

            if matches is None or len(matches) < self.min_matches:
                print(f"  图像 {i+1} 匹配点不足，使用简单水平拼接")
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue

            homography_result = self.find_homography(kp1, kp2, matches)

            if homography_result is None:
                print(f"  无法计算单应性矩阵，使用简单水平拼接")
                h = max(result.shape[0], images[i].shape[0])
                w = result.shape[1] + images[i].shape[1]
                new_img = np.zeros((h, w, 3), dtype=np.uint8)
                new_img[:result.shape[0], :result.shape[1]] = result
                new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                result = new_img
                continue

            H, inlier_ratio, _, _ = homography_result

            print(f"  匹配点: {len(matches)}, 内点率: {inlier_ratio:.2%}")

            if abs(H[0, 0] - 1.0) < 0.1 and abs(H[1, 1] - 1.0) < 0.1 and abs(H[0, 1]) < 0.1 and abs(H[1, 0]) < 0.1:
                print("  检测到主要是平移变换，使用优化拼接")

                dx = H[0, 2]
                dy = H[1, 2]

                h1, w1 = result.shape[:2]
                h2, w2 = images[i].shape[:2]

                corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
                corners2 = np.float32([[dx, dy], [w2 + dx, dy],
                                      [w2 + dx, h2 + dy], [dx, h2 + dy]])

                all_corners = np.vstack([corners1, corners2])

                x_min = int(np.floor(all_corners[:, 0].min()))
                x_max = int(np.ceil(all_corners[:, 0].max()))
                y_min = int(np.floor(all_corners[:, 1].min()))
                y_max = int(np.ceil(all_corners[:, 1].max()))

                tx = -x_min if x_min < 0 else 0
                ty = -y_min if y_min < 0 else 0

                new_width = max(x_max - x_min, 1)
                new_height = max(y_max - y_min, 1)

                canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                mask = np.zeros((new_height, new_width), dtype=np.uint8)

                canvas[ty:ty+h1, tx:tx+w1] = result
                mask[ty:ty+h1, tx:tx+w1] = 255

                img2_x = int(round(tx + dx))
                img2_y = int(round(ty + dy))

                y1_start = max(0, img2_y)
                y1_end = min(new_height, img2_y + h2)
                x1_start = max(0, img2_x)
                x1_end = min(new_width, img2_x + w2)

                y2_start = max(0, -img2_y)
                y2_end = y2_start + (y1_end - y1_start)
                x2_start = max(0, -img2_x)
                x2_end = x2_start + (x1_end - x1_start)

                if y1_end > y1_start and x1_end > x1_start:
                    overlap_mask = mask[y1_start:y1_end, x1_start:x1_end]

                    if np.any(overlap_mask > 0):
                        existing_region = canvas[y1_start:y1_end, x1_start:x1_end]
                        new_region = images[i][y2_start:y2_end, x2_start:x2_end]

                        dist1 = cv2.distanceTransform((overlap_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
                        new_mask = np.ones((y1_end - y1_start, x1_end - x1_start), dtype=np.uint8) * 255
                        dist2 = cv2.distanceTransform(new_mask, cv2.DIST_L2, 5)

                        weight1 = dist1 / (dist1 + dist2 + 1e-10)
                        weight2 = 1.0 - weight1

                        weight1 = np.stack([weight1] * 3, axis=2)
                        weight2 = np.stack([weight2] * 3, axis=2)

                        blended = (existing_region * weight1 + new_region * weight2).astype(np.uint8)
                        blended = np.where(overlap_mask[:,:,np.newaxis] > 0, blended, new_region)

                        canvas[y1_start:y1_end, x1_start:x1_end] = blended
                        mask[y1_start:y1_end, x1_start:x1_end] = 255
                    else:
                        canvas[y1_start:y1_end, x1_start:x1_end] = images[i][y2_start:y2_end, x2_start:x2_end]
                        mask[y1_start:y1_end, x1_start:x1_end] = 255

                    result = canvas
                    print(f"   成功拼接图像，当前尺寸: {result.shape}")
                else:
                    print(f"   计算的放置区域无效，使用一般拼接方法")
                    new_result = self.stitch_pair(result, images[i], H)
                    if new_result is not None:
                        result = new_result
            else:
                new_result = self.stitch_pair(result, images[i], H)
                if new_result is not None:
                    result = new_result
                else:
                    print(f"  图像 {i+1} 拼接失败，使用简单水平拼接")
                    h = max(result.shape[0], images[i].shape[0])
                    w = result.shape[1] + images[i].shape[1]
                    new_img = np.zeros((h, w, 3), dtype=np.uint8)
                    new_img[:result.shape[0], :result.shape[1]] = result
                    new_img[:images[i].shape[0], result.shape[1]:] = images[i]
                    result = new_img

            print(f"  当前结果尺寸: {result.shape}")

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

        h, w = gray.shape
        template_size = min(100, w//10)

        if template_size < 20:
            return stitched_image

        template = gray[:, :template_size]

        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

        loc = np.where(res >= threshold)

        if len(loc[0]) > 0 and len(loc[1]) > 0:
            first_match = loc[1].min()

            if first_match > template_size and first_match < w - template_size:
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
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        processed.append(denoised)

    return processed

def main():
    IMAGE_FOLDER = "fp"

    print("=" * 50)
    print("加载图像...")
    images = load_images(IMAGE_FOLDER)

    if len(images) < 2:
        print("需要至少2张图像进行拼接")
        return

    print(f"成功加载 {len(images)} 张图像")

    print("\n" + "=" * 50)
    print("预处理图像...")
    processed_images = preprocess_images(images)

    print("\n" + "=" * 50)
    print("参数配置说明:")
    print("  min_matches: 最少匹配点数 (默认15，越高越严格)")
    print("  ransac_thresh: RANSAC阈值 (默认3.0，越小越严格)")
    print("\n如果拼接失败，可以尝试调整这两个参数:")
    print("  - 减小 min_matches (如10) 以允许更弱的匹配")
    print("  - 增大 ransac_thresh (如5.0) 以提高容错性")

    stitcher = CylinderImageStitcher(min_matches=15, ransac_thresh=3.0)

    print("\n" + "=" * 50)
    print("执行全局拼接...")
    result_global = stitcher.stitch_all_pairs_fixed(processed_images)

    print("\n" + "=" * 50)
    print("去除重复区域...")

    if result_global is not None:
        result_global_clean = remove_duplicate_overlap(result_global)
        cv2.imwrite("stitched_result.jpg", result_global_clean)
        print(f"拼接结果已保存为: stitched_result.jpg")
        print(f"输出尺寸: {result_global_clean.shape}")

        print("\n" + "=" * 50)
        print("显示拼接结果...")
        show_images_with_opencv([result_global_clean], ["拼接结果"], "拼接结果")
    else:
        print("拼接失败！")
        print("\n建议尝试:")
        print("  1. 检查图像是否有足够特征点（纹理丰富）")
        print("  2. 减小 min_matches 参数到10或更低")
        print("  3. 增大 ransac_thresh 参数到5.0或更高")
        print("  4. 确保图像之间有足够的重叠区域（至少20%）")

    print("\n" + "=" * 50)
    print("拼接完成！")

if __name__ == "__main__":
    main()
