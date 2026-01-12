import threading
import time
import cv2
import serial
import datetime
import queue
import sys
import os
from typing import Optional

class MultiThreadCameraApp:
    def __init__(self):
        # 串口相关变量
        self.serial_port: Optional[serial.Serial] = None
        self.serial_thread_running = False
        self.serial_queue = queue.Queue()
        
        # 摄像头相关变量
        self.camera = None
        self.camera_lock = threading.Lock()
        self.last_capture_time = 0
        self.capture_count = 0
        
        # 照片计数
        self.num = 0
        # 线程控制
        self.running = True
        
        # 串口配置 (根据您的实际情况修改)
        self.serial_config = {
            'port': 'COM17',  # Windows: COM3, Linux: /dev/ttyUSB0
            'baudrate': 9600,
            'timeout': 1
        }
        
        # 摄像头配置
        self.camera_config = {
            'device_id': 1,  # 默认摄像头
            'save_dir': 'captures',
            'prefix': 'capture'
        }
        
        # 创建保存目录
        if not os.path.exists(self.camera_config['save_dir']):
            os.makedirs(self.camera_config['save_dir'])

    def serial_listener_thread(self):
        """串口监听线程 - 监听并发送相同数据"""
        print(f"串口监听线程启动，尝试打开串口: {self.serial_config['port']}")
        
        try:
            self.serial_port = serial.Serial(
                port=self.serial_config['port'],
                baudrate=self.serial_config['baudrate'],
                timeout=self.serial_config['timeout']
            )
            print(f"串口 {self.serial_config['port']} 已成功打开")
        except Exception as e:
            print(f"无法打开串口 {self.serial_config['port']}: {e}")
            self.serial_port = None
        
        self.serial_thread_running = True
        
        while self.running and self.serial_thread_running:
            try:
                if self.serial_port and self.serial_port.is_open:
                    # 读取串口数据
                    if self.serial_port.in_waiting > 0:
                        data = self.serial_port.read(self.serial_port.in_waiting)
                        if data:
                            # 打印接收到的数据
                            received_str = data.decode('utf-8', errors='ignore').strip()
                            if received_str=="1":
                                print(f"[收到拍摄指令] {received_str}")
                                self.capture_image(trigger_source="Serial",file_name=str(self.num+1))
                                self.num += 1
                            time.sleep(1)

                            # 发送相同的数据回串口
                            self.serial_port.write(("ok").encode('utf-8'))
                            print(f"[串口发送] 已回发2")
                
                time.sleep(0.1)  # 避免CPU占用过高
                
            except Exception as e:
                print(f"串口线程错误: {e}")
                time.sleep(1)
        
        # 关闭串口
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        print("串口监听线程已停止")

    def console_print_thread(self):
        """控制台打印线程 - 打印hello并控制摄像头拍摄"""
        print("控制台打印线程启动")
        hello_count = 0
        
        while self.running:
            try:
                # 打印hello
                hello_count += 1
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[线程2] {current_time} - running waiting for stiched #{hello_count}")
                
                time.sleep(2)  # 每2秒打印一次
                
            except Exception as e:
                print(f"控制台打印线程错误: {e}")
                time.sleep(1)
        
        print("控制台打印线程已停止")

    # def camera_timer_thread(self):
    #     """定时拍摄线程 - 每10秒拍摄一次"""
    #     print("定时拍摄线程启动")
        
    #     while self.running:
    #         try:
    #             current_time = time.time()
                
    #             # 每10秒拍摄一次
    #             if current_time - self.last_capture_time >= 10:
    #                 self.last_capture_time = current_time
    #                 print(f"[线程3] 定时拍摄触发 ({datetime.datetime.now().strftime('%H:%M:%S')})")
    #                 self.capture_image("scheduled")
                
    #             time.sleep(1)  # 每秒检查一次
                
    #         except Exception as e:
    #             print(f"定时拍摄线程错误: {e}")
    #             time.sleep(1)
        
    #     print("定时拍摄线程已停止")

    def capture_image(self, trigger_source="manual", file_name="name"):
        """拍摄图像并保存"""
        with self.camera_lock:
            try:
                # 初始化摄像头（如果未初始化）
                if self.camera is None:
                    self.camera = cv2.VideoCapture(self.camera_config['device_id'])
                    if not self.camera.isOpened():
                        print("无法打开摄像头")
                        return
                
                # 读取一帧
                ret, frame = self.camera.read()
                
                if ret:
                    self.capture_count += 1
                    
                    # 生成文件名
                    filename = f"{file_name}.jpg"
                    filepath = os.path.join(self.camera_config['save_dir'], filename)
                    
                    # 保存图像
                    cv2.imwrite(filepath, frame)
                    print(f"[摄像头] 已保存图像: {filename}")
                    
                    
                    # 显示图像预览（可选）
                    # preview_frame = cv2.resize(frame, (320, 240))
                    # cv2.imshow(f'Capture Preview - {trigger_source}', preview_frame)
                    # cv2.waitKey(500)  # 显示500ms
                    # cv2.destroyAllWindows()
                    
                else:
                    print("[摄像头] 无法读取帧")
                    
            except Exception as e:
                print(f"[摄像头] 拍摄错误: {e}")

    def cleanup(self):
        """清理资源"""
        print("\n正在停止所有线程...")
        self.running = False
        self.serial_thread_running = False
        
        # 等待线程结束
        time.sleep(1)
        
        # 释放摄像头
        if self.camera:
            self.camera.release()
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        print("程序已停止")

    def run(self):
        """主运行函数"""
        print("=" * 50)
        print("多线程摄像头应用程序")
        print("=" * 50)
        print("线程1: 串口监听与数据回发")
        print("线程2: 控制台打印Hello并手动触发拍摄")
        print("线程3: 定时拍摄（每10秒一次）")
        print("=" * 50)
        print("按 Ctrl+C 退出程序")
        print("=" * 50)
        
        # 创建并启动线程
        threads = []
        
        # 线程1: 串口监听
        serial_thread = threading.Thread(target=self.serial_listener_thread, name="SerialThread")
        threads.append(serial_thread)
        
        # 线程2: 控制台打印
        console_thread = threading.Thread(target=self.console_print_thread, name="ConsoleThread")
        threads.append(console_thread)
        
        # 线程3: 定时拍摄
        # camera_thread = threading.Thread(target=self.camera_timer_thread, name="CameraTimerThread")
        # threads.append(camera_thread)
        
        # 启动所有线程
        for thread in threads:
            thread.daemon = True  # 设置为守护线程
            thread.start()
            print(f"已启动线程: {thread.name}")
        
        # 等待所有线程
        try:
            while any(thread.is_alive() for thread in threads):
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n接收到中断信号，正在退出...")
        finally:
            self.cleanup()

def main():
    """主函数"""
    app = MultiThreadCameraApp()
    
    # 显示使用说明
    print("\n使用说明:")
    print("1. 修改 serial_config 中的端口号以匹配您的串口设备")
    print("2. 如果没有串口设备，程序将使用模拟模式")
    print("3. 拍摄的图像将保存在 'captures' 文件夹中")
    print("4. 确保摄像头设备可用")
    
    # 检查摄像头
    test_camera = cv2.VideoCapture(0)
    if test_camera.isOpened():
        print("✓ 摄像头检测成功")
        test_camera.release()
    else:
        print("⚠ 无法访问摄像头，请检查摄像头连接")
    
    # 运行应用程序
    input("\n按 Enter 键开始运行...")
    app.run()

if __name__ == "__main__":
    # 安装所需库的提示
    required_libraries = ['opencv-python', 'pyserial']
    print("需要的库:")
    for lib in required_libraries:
        print(f"  pip install {lib}")
    
    # 检查opencv
    try:
        import cv2
    except ImportError:
        print("\n错误: 未找到OpenCV，请安装:")
        print("pip install opencv-python")
        sys.exit(1)
    
    # 检查pyserial
    try:
        import serial
    except ImportError:
        print("\n警告: 未找到pyserial，串口功能将使用模拟模式")
        print("如需使用真实串口，请安装:")
        print("pip install pyserial")
    
    main()