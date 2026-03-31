import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import time
import sys

class UR10SmartPose(Node):
    def __init__(self):
        super().__init__('ur10_smart_pose_node')
        # 话题名必须与 controllers.yaml 一致
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.get_logger().info('🤖 节点启动...')

    def move_to_target(self):
        # --- 1. 检查是否有控制器在听 (关键调试步骤) ---
        timeout = 5 # 等待 5 秒
        start_time = time.time()
        while self.publisher_.get_subscription_count() == 0:
            if time.time() - start_time > timeout:
                self.get_logger().error('❌ 错误：没有控制器订阅此话题！')
                self.get_logger().error('   请检查: 1. 仿真是否开启 2. 控制器是否激活')
                self.get_logger().error('   尝试运行: ros2 control list_controllers')
                return
            self.get_logger().warn('⏳ 等待控制器连接...')
            time.sleep(0.5)
        
        self.get_logger().info('✅ 控制器已连接！准备发送指令...')

        msg = JointTrajectory()
        msg.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        point = JointTrajectoryPoint()

        # ================= 更新后的目标角度 =================
        # 角度直接取自提供的最新目标状态图（第二张图）
        degrees = [
            -16.0,  # shoulder_pan
            -92.0,  # shoulder_lift
            52.0,   # elbow
            -48.0,  # wrist_1
            -90.0,  # wrist_2
            -14.0   # wrist_3
        ]
        
        radians = [math.radians(d) for d in degrees]
        print(f"🎯 目标角度: {degrees}")
        print(f"📐 发送弧度: {[round(r, 2) for r in radians]}")
        
        point.positions = radians
        # 设定 6 秒的移动时间，保证平稳到达目标点
        point.time_from_start = Duration(sec=6, nanosec=0)
        
        msg.points.append(point)
        self.publisher_.publish(msg)
        self.get_logger().info('🚀 指令已发送！')

def main(args=None):
    rclpy.init(args=sys.argv)
    node = UR10SmartPose()
    node.move_to_target()
    time.sleep(2)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
