import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional

class ImageStitcher:
    def __init__(self, images_per_row: int = 5):
        """
        初始化图片拼接器
        
        Args:
            images_per_row: 每行显示的图片数量，默认为5
        """
        self.images_per_row = images_per_row
    
    def load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        加载图片列表
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            加载的图片数组列表
        """
        images = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"警告：图片不存在 - {img_path}")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图片 - {img_path}")
                continue
                
            images.append(img)
        return images
    
    def resize_images(self, images: List[np.ndarray], target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        调整图片到相同大小
        
        Args:
            images: 图片列表
            target_size: 目标尺寸 (宽度, 高度)，如果为None则使用第一张图片的尺寸
            
        Returns:
            调整后的图片列表
        """
        if not images:
            return []
            
        if target_size is None:
            # 使用第一张图片的尺寸
            target_size = (images[0].shape[1], images[0].shape[0])
        
        resized_images = []
        for img in images:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized_images.append(resized_img)
            
        return resized_images
    
    def stitch_images(self, images: List[np.ndarray], 
                     images_per_row: Optional[int] = None,
                     padding: int = 0,
                     background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        拼接图片
        
        Args:
            images: 图片列表
            images_per_row: 每行图片数量，如果为None则使用初始化值
            padding: 图片之间的间距
            background_color: 背景颜色 (B, G, R)
            
        Returns:
            拼接后的大图
        """
        if not images:
            raise ValueError("没有可用的图片进行拼接")
        
        if images_per_row is None:
            images_per_row = self.images_per_row
        
        # 获取图片尺寸
        img_height, img_width = images[0].shape[:2]
        channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
        
        # 计算行数和列数
        num_images = len(images)
        rows = (num_images + images_per_row - 1) // images_per_row  # 向上取整
        
        # 计算大图尺寸
        total_width = img_width * images_per_row + padding * (images_per_row + 1)
        total_height = img_height * rows + padding * (rows + 1)
        
        # 创建空白大图
        if channels == 1:
            stitched_image = np.full((total_height, total_width), background_color[0], dtype=np.uint8)
        else:
            stitched_image = np.full((total_height, total_width, 3), background_color, dtype=np.uint8)
        
        # 将图片粘贴到大图中
        for i, img in enumerate(images):
            row = i // images_per_row
            col = i % images_per_row
            
            # 计算粘贴位置
            x_start = padding + col * (img_width + padding)
            y_start = padding + row * (img_height + padding)
            x_end = x_start + img_width
            y_end = y_start + img_height
            
            # 如果图片通道数不匹配，进行转换
            if len(img.shape) != len(stitched_image.shape):
                if len(img.shape) == 2 and len(stitched_image.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif len(img.shape) == 3 and len(stitched_image.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 粘贴图片
            stitched_image[y_start:y_end, x_start:x_end] = img
            
            # 可选：添加图片编号
            self.add_image_number(stitched_image, i+1, (x_start, y_start - 5))
        
        return stitched_image
    
    def add_image_number(self, image: np.ndarray, number: int, position: Tuple[int, int]):
        """
        在图片上添加编号
        
        Args:
            image: 图片
            number: 编号
            position: 位置 (x, y)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 0, 255)  # 红色
        thickness = 2
        
        text = f"{number}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 调整位置确保文本可见
        x, y = position
        if y < text_size[1]:
            y = text_size[1] + 5
        
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    
    def stitch_from_folder(self, folder_path: str, 
                          image_extensions: List[str] = None,
                          target_size: Optional[Tuple[int, int]] = None,
                          **kwargs) -> np.ndarray:
        """
        从文件夹加载图片并拼接
        
        Args:
            folder_path: 文件夹路径
            image_extensions: 图片扩展名列表，默认为 ['.jpg', '.jpeg', '.png', '.bmp']
            target_size: 目标尺寸
            **kwargs: 传递给stitch_images的参数
            
        Returns:
            拼接后的大图
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # 获取所有图片文件
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"文件夹不存在: {folder_path}")
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(folder.glob(f'*{ext}')))
            image_paths.extend(list(folder.glob(f'*{ext.upper()}')))
        
        if not image_paths:
            raise ValueError(f"在文件夹中未找到图片: {folder_path}")
        
        # 按文件名排序
        image_paths = sorted(image_paths, key=lambda x: x.name)
        
        # 加载图片
        images = self.load_images([str(path) for path in image_paths])
        
        # 调整图片大小
        if target_size is not None or len(images) > 0:
            images = self.resize_images(images, target_size)
        
        # 拼接图片
        return self.stitch_images(images, **kwargs)


def main():
    """主函数示例"""
    
    # 示例1：从文件夹加载图片并拼接
    print("示例1：从文件夹加载图片")
    stitcher = ImageStitcher(images_per_row=5)
    
    try:
        # 假设有一个名为'images'的文件夹，里面包含要拼接的图片
        result = stitcher.stitch_from_folder(
            folder_path='assets',
            padding=0,
            background_color=(240, 240, 240)  # 浅灰色背景
        )
        
        # 保存结果
        cv2.imwrite('stitched_result.jpg', result)
        print(f"拼接完成，结果已保存为 'stitched_result.jpg'")
        print(f"结果图片尺寸: {result.shape[1]}x{result.shape[0]}")
        
        # 显示结果
        cv2.imshow('Stitched Image', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except ValueError as e:
        print(f"错误: {e}")
        print("创建示例图片...")
        create_sample_images()
        
        # 重新尝试拼接
        result = stitcher.stitch_from_folder(
            folder_path='sample_images',
            padding=15,
            background_color=(240, 240, 240)
        )
        cv2.imwrite('stitched_result.jpg', result)
        print(f"使用示例图片拼接完成，结果已保存为 'stitched_result.jpg'")
    
    # 示例2：直接使用图片列表
    print("\n示例2：使用图片列表")
    # 创建一些示例图片
    sample_images = []
    for i in range(12):
        # 创建不同颜色的图片
        color = [(i*20) % 255, (i*40) % 255, (i*60) % 255]
        img = np.full((150, 200, 3), color, dtype=np.uint8)
        
        # 添加文字
        cv2.putText(img, f'Image {i+1}', (50, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        sample_images.append(img)
    
    # 拼接示例图片
    stitcher2 = ImageStitcher(images_per_row=5)
    result2 = stitcher2.stitch_images(
        sample_images,
        padding=0,
        background_color=(200, 200, 200)
    )
    
    # 保存结果
    cv2.imwrite('stitched_sample.jpg', result2)
    print(f"示例图片拼接完成，结果已保存为 'stitched_sample.jpg'")
    
    # 显示结果
    cv2.imshow('Stitched Sample', result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_sample_images():
    """创建示例图片文件夹"""
    os.makedirs('sample_images', exist_ok=True)
    
    for i in range(12):
        # 创建不同颜色的图片
        color = [(i*20) % 255, (i*40) % 255, (i*60) % 255]
        img = np.full((150, 200, 3), color, dtype=np.uint8)
        
        # 添加文字
        cv2.putText(img, f'Sample {i+1}', (50, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存图片
        cv2.imwrite(f'sample_images/sample_{i+1:02d}.jpg', img)
    
    print("已创建12张示例图片到 'sample_images' 文件夹")


if __name__ == "__main__":
    main()