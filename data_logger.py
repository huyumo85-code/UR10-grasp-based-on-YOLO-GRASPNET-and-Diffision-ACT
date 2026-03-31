import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import datetime
import zarr

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        self.bridge = CvBridge()
        self.base_path = os.path.expanduser("~/workspaces/ur_gz/robot_diffusion_data/zarr_data")
        os.makedirs(self.base_path, exist_ok=True)

        # 订阅话题
        self.create_subscription(Image, '/d435/image', self.hand_eye_cb, 10)
        self.create_subscription(Image, '/camera_global/image', self.global_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        
        # 🌟 监听抓取管理器的自动控制信号
        self.create_subscription(String, '/record_signal', self.signal_cb, 10)

        self.img_hand_eye = None
        self.img_global = None
        self.current_joints = None
        self.is_recording = False
        self.episode_data = []

        # 🌟 严格锁定 7 维关节顺序
        self.target_joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
            'robotiq_85_left_knuckle_joint'
        ]

        # 🎯 DP 标准采样率：10Hz
        self.sample_hz = 10.0
        self.sample_timer = self.create_timer(1.0 / self.sample_hz, self.tick)
        
        self.create_timer(1.0, self.show_status)
        self.get_logger().info(f'🚀 全自动分段录制机已启动 ({self.sample_hz}Hz)，等待控制端信号...')

    def signal_cb(self, msg):
        cmd = msg.data
        if cmd == 'START':
            self.episode_data = []
            self.is_recording = True
            print("\n" + "🎬"*20)
            self.get_logger().info("🟢 收到 [START] 信号，开始录制当前轨迹！")
        elif cmd == 'END':
            if self.is_recording:
                self.is_recording = False
                self.get_logger().info("🔴 收到 [END] 信号，正在保存数据段...")
                self.save_episode()
        elif cmd == 'ABORT':
            if self.is_recording:
                self.is_recording = False
                self.episode_data = []
                self.get_logger().warn("❌ 收到 [ABORT] 信号，已丢弃本次残次数据！")

    def hand_eye_cb(self, msg):
        self.img_hand_eye = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def global_cb(self, msg):
        self.img_global = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def joint_cb(self, msg):
        pos_dict = dict(zip(msg.name, msg.position))
        try:
            self.current_joints = [pos_dict[name] for name in self.target_joints]
        except KeyError:
            pass 

    def tick(self):
        if self.is_recording:
            if self.img_hand_eye is not None and self.img_global is not None and self.current_joints is not None:
                # 🌟 录制时直接缩放为 224x224，极大减小单个 demo 的体积
                img_h_resized = cv2.resize(self.img_hand_eye, (224, 224))
                img_g_resized = cv2.resize(self.img_global, (224, 224))
                
                self.episode_data.append({
                    'img_hand_eye': img_h_resized,
                    'img_global': img_g_resized,
                    'joints': self.current_joints
                })

    def show_status(self):
        if self.is_recording:
            count = len(self.episode_data)
            print(f"\r🎥 录制中 | 已采集: {count:3d} 帧 | 时长: {count/self.sample_hz:.1f}s", end="")

    def save_episode(self):
        if len(self.episode_data) < 10:
            print("\n⚠️ 轨迹过短，自动丢弃。")
            self.episode_data = []
            return

        ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
        zarr_path = os.path.join(self.base_path, f"demo_{ts}.zarr")
        
        root = zarr.open(zarr_path, mode='w')
        
        imgs_hand = np.array([d['img_hand_eye'] for d in self.episode_data], dtype=np.uint8)
        imgs_global = np.array([d['img_global'] for d in self.episode_data], dtype=np.uint8)
        joints = np.array([d['joints'] for d in self.episode_data], dtype=np.float32)

        # 启用压缩
        compressor = zarr.Blosc(cname='zstd', clevel=3)
        root.create_dataset('data/obs/img_hand_eye', data=imgs_hand, chunks=(100, 224, 224, 3), compressor=compressor)
        root.create_dataset('data/obs/img_global', data=imgs_global, chunks=(100, 224, 224, 3), compressor=compressor)
        root.create_dataset('data/action', data=joints, chunks=(1000, 7))
        
        print(f"\n✅ 数据段保存成功: {zarr_path}")
        self.episode_data = []

def main():
    rclpy.init()
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
