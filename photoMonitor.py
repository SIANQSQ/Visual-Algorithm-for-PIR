import os
import time
import re
import threading
from pathlib import Path
from collections import defaultdict

class ImageMonitor:
    def __init__(self, folder_path):
        """
        初始化图片监控器
        
        Args:
            folder_path: 要监控的文件夹路径
        """
        self.folder_path = Path(folder_path)
        self.image_groups = defaultdict(list)  # 存储按前缀分组的图片
        self.processed_images = set()  # 存储已处理的图片，避免重复处理
        self.running = True
        self.image_pattern = re.compile(r'^(\d+)-([1-5])\.jpg$')
        
    def extract_prefix_and_number(self, filename):
        """
        从文件名中提取前缀和序号
        
        Args:
            filename: 文件名
            
        Returns:
            (前缀, 序号) 或 (None, None) 如果格式不匹配
        """
        match = self.image_pattern.match(filename)
        if match:
            prefix = match.group(1)  # 前缀数字
            number = int(match.group(2))  # 序号 (1-5)
            return prefix, number
        return None, None
    
    def scan_folder(self):
        """
        扫描文件夹，处理新出现的图片
        
        Returns:
            是否有新的完整组出现
        """
        has_new_group = False
        
        # 获取所有jpg文件
        jpg_files = list(self.folder_path.glob('*.jpg'))
        
        for file_path in jpg_files:
            filename = file_path.name
            
            # 跳过已处理的图片
            if filename in self.processed_images:
                continue
                
            prefix, number = self.extract_prefix_and_number(filename)
            
            if prefix and number:
                # 添加到对应前缀的组中
                self.image_groups[prefix].append((number, str(file_path)))
                self.processed_images.add(filename)
                print(f"已添加图片: {filename} (前缀: {prefix}, 序号: {number})")
                
                # 检查是否凑齐一组
                if len(self.image_groups[prefix]) == 5:
                    has_new_group = True
                    self.process_complete_group(prefix)
        
        return has_new_group
    
    def process_complete_group(self, prefix):
        """
        处理一个完整的图片组（已凑齐5张）
        
        Args:
            prefix: 图片前缀
        """
        print(f"\n{'='*50}")
        print(f"发现完整图片组! 前缀: {prefix}")
        
        # 获取该组的所有图片（按序号排序）
        group_images = sorted(self.image_groups[prefix], key=lambda x: x[0])
        
        # 提取图片路径列表（按顺序）
        image_paths = [img[1] for img in group_images]
        
        # 这里可以调用你的拼接算法
        # 例如: result = stitch_images(image_paths)
        
        print(f"图片路径列表 (按顺序):")
        for i, (num, path) in enumerate(group_images, 1):
            print(f"  图片{i}: {os.path.basename(path)} -> {path}")
        
        print(f"前缀 {prefix} 已保存")
        print(f"{'='*50}\n")
        
        # 处理完成后从字典中移除（可选）
        # 如果不移除，后续可能会重复处理同一前缀的图片
        # self.image_groups.pop(prefix, None)
        
        # 或者只是标记为已处理（推荐）
        self.image_groups[prefix] = []
    
    def monitor(self, interval=1):
        """
        持续监控文件夹
        
        Args:
            interval: 扫描间隔（秒）
        """
        print(f"开始监控文件夹: {self.folder_path}")
        print(f"图片命名格式应为: xxx-x.jpg (xxx是数字前缀，x是1-5的序号)")
        print("按Ctrl+C或输入'stop'停止监控\n")
        
        while self.running:
            try:
                self.scan_folder()
                
                # 显示当前状态
                active_groups = {k: len(v) for k, v in self.image_groups.items() if v}
                if active_groups:
                    print(f"当前活跃组: {active_groups}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n收到中断信号，停止监控...")
                self.stop()
            except Exception as e:
                print(f"监控过程中出现错误: {e}")
    
    def stop(self):
        """停止监控"""
        self.running = False
        print("监控已停止")
        print(f"总共处理了 {len(self.processed_images)} 张图片")

def input_monitor(image_monitor):
    """独立的输入监控线程，用于接收停止命令"""
    while image_monitor.running:
        user_input = input()
        if user_input.lower() == 'stop':
            print("收到停止命令...")
            image_monitor.stop()
            break

def main():
    """主函数"""

    folder_path = "./captures"
    
    # 确保文件夹存在
    folder = Path(folder_path)
    if not folder.exists():
        print(f"文件夹 '{folder_path}' 不存在，是否创建？(y/n): ", end="")
        if input().lower() == 'y':
            folder.mkdir(parents=True, exist_ok=True)
            print(f"已创建文件夹: {folder_path}")
        else:
            print("程序退出")
            return
    
    # 创建监控器
    monitor = ImageMonitor(folder_path)
    
    # 启动输入监控线程
    input_thread = threading.Thread(target=input_monitor, args=(monitor,))
    input_thread.daemon = True
    input_thread.start()
    
    try:
        # 开始监控
        monitor.monitor(interval=0.5)  # 每0.5秒扫描一次
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("\n程序结束")
        print(f"总计处理图片: {len(monitor.processed_images)}")

if __name__ == "__main__":
    main()