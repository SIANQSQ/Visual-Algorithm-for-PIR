import cv2
import numpy as np

def unwarp_cylindrical(src_img, focal_length_px, rotation_angle=0):
    """
    将拍摄的圆弧面图像展开为矩形图。
    
    参数:
        src_img: 输入的圆弧面图像（BGR格式）。
        focal_length_px: 相机焦距（像素单位）。这是最重要的参数，可通过相机标定获取，
                         或近似为：focal_length_px = (image_width / 2) / tan(FOV_horizontal / 2)。
        rotation_angle: 相机光轴与圆柱法线的偏转角（弧度），通常很小，默认为0。
    
    返回:
        unwarped_img: 展开后的矩形图像。
        mask: 有效区域掩码。
    """
    h, w = src_img.shape[:2]
    
    # 1. 计算图像中心点（假设主点位于图像中心）
    cx = w / 2.0
    cy = h / 2.0
    
    # 2. 创建展开图的坐标网格
    # 展开图的宽度可对应圆柱的周长，高度与圆柱高一致
    # 这里假设展开360度（2π弧度），可根据实际圆弧角度调整
    unwarp_w = int(2 * np.pi * focal_length_px)  # 近似周长
    unwarp_h = h
    unwarped_img = np.zeros((unwarp_h, unwarp_w, 3), dtype=np.uint8)
    
    # 创建目标图的x', y'坐标矩阵 (meshgrid)
    x_prime = np.tile(np.arange(unwarp_w), (unwarp_h, 1)).astype(np.float32)
    y_prime = np.repeat(np.arange(unwarp_h).reshape(-1, 1), unwarp_w, axis=1).astype(np.float32)
    
    # 3. 核心：根据柱面反变换公式，计算原图坐标
    # 公式推导：x' = f * θ, y' = f * h / sqrt(f^2 + (x-cx)^2) * (y-cy) + cy
    # 反向求解：θ = x' / f
    #          x = f * tan(θ) + cx
    #          y = (y' - cy) * sqrt(f^2 + (x-cx)^2) / f + cy
    theta = (x_prime - unwarp_w / 2.0) / focal_length_px
    x = focal_length_px * np.tan(theta) + cx
    y = (y_prime - cy) * np.sqrt(focal_length_px**2 + (x - cx)**2) / focal_length_px + cy
    
    # 4. 使用remap进行高效的重采样（双线性插值）
    map_x = x.astype(np.float32)
    map_y = y.astype(np.float32)
    unwarped_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # 5. 生成有效区域掩码（可选，用于后续拼接）
    mask = np.ones((unwarp_h, unwarp_w), dtype=np.uint8) * 255
    # 可在此处根据变换后坐标的有效性调整mask
    
    return unwarped_img, mask

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 1. 读取拍摄的圆弧面图像
    src_image = cv2.imread('input.jpg')
    if src_image is None:
        print("input.jpg' 存在。")
        exit()
    
    # 2. 关键：设置焦距（像素单位）
    # 示例值，必须根据你的实际情况调整！
    # 获取方法1：相机标定（最准确）
    # 获取方法2：估算。例如，已知相机水平视场角FOV=60度，图像宽w=1920像素，
    # 则 focal_length_px ≈ (w/2) / tan(np.radians(FOV/2)) ≈ (1920/2) / tan(30°) ≈ 1663
    FOCAL_LENGTH_ESTIMATE = 1000.0  # 请修改此值！
    
    # 3. 执行展开
    unwarped, mask = unwarp_cylindrical(src_image, FOCAL_LENGTH_ESTIMATE)
    
    # 4. 显示并保存结果
    cv2.imshow('Original Arc Image', src_image)
    cv2.imshow('Unwarped Rectangular Image', unwarped)
    cv2.imwrite('unwarped_result.jpg', unwarped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()