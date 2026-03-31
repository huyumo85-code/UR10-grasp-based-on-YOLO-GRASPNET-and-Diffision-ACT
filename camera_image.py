import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np # 👈 用于图像拼接
import sys

class DualCameraReader(Node):
    def __init__(self):
        super().__init__('dual_camera_reader_node')
        
        self.bridge = CvBridge()
        
        # 定义 QoS 策略：必须匹配你 Bridge 里的 best_effort
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # 1. 订阅手眼相机 (D435)
        self.sub_hand = self.create_subscription(
            Image, 
            '/d435/image', 
            self.hand_camera_callback, 
            qos_profile
        )
        
        # 2. 订阅全局相机 (Global)
        self.sub_global = self.create_subscription(
            Image, 
            '/camera_global/image', 
            self.global_camera_callback, 
            qos_profile
        )

        # 存储最新的图像帧
        self.cv_img_hand = None
        self.cv_img_global = None
        
        self.get_logger().info('🚀 双相机节点启动！正在接收 /d435 和 /camera_global 数据...')

    def hand_camera_callback(self, msg):
        try:
            self.cv_img_hand = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.display_images()
        except Exception as e:
            self.get_logger().error(f'手眼相机转换失败: {e}')

    def global_camera_callback(self, msg):
        try:
            self.cv_img_global = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.display_images()
        except Exception as e:
            self.get_logger().error(f'全局相机转换失败: {e}')

    def display_images(self):
        # 只有当两个相机的画面都收到过一次后，才开始显示
        if self.cv_img_hand is not None and self.cv_img_global is not None:
            
            # 在画面上加标注以便区分
            img_hand = self.cv_img_hand.copy()
            img_global = self.cv_img_global.copy()
            
            cv2.putText(img_hand, "Hand Camera (D435)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_global, "Global Camera", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # --- 方案 A: 拼接显示 (推荐) ---
            # 确保两张图尺寸一致（如果 Gazebo 设置的都是 640x480 则直接拼接）
            if img_hand.shape == img_global.shape:
                combined_view = np.hstack((img_hand, img_global)) # 水平拼接
                cv2.imshow("Dual Camera Monitoring", combined_view)
            else:
                # 如果尺寸不同，分两个窗口显示
                cv2.imshow("Hand Camera View", img_hand)
                cv2.imshow("Global Camera View", img_global)

            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=sys.argv)
    node = DualCameraReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
