import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor # 新增：解决死锁的关键
import tf2_ros
import numpy as np
import requests
import json
import os
import threading
import time
import cv2
import message_filters

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped
from control_msgs.action import GripperCommand
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from control_msgs.msg import GripperCommand as GripperMsg

# ================= 配置区域 =================
SERVICE_URL = "http://localhost:5005/predict"
ARM_GROUP_NAME = "ur_manipulator"
TCP_OFFSET_Z = 0.112
PRE_GRASP_OFFSET = 0.10
LIFT_HEIGHT = 0.15
MIN_SAFE_Z = 0.48

TEMP_RGB = "/tmp/ros_rgb.png"
TEMP_DEPTH = "/tmp/ros_depth.png"
VIS_PATH = "/tmp/ros_vis.png" # 固定的可视化图片路径
HOME_JOINTS = [-0.28, -1.61, 0.91, -0.84, -1.57, -0.24]
# ===========================================

def quat2mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, 0],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0],
        [0, 0, 0, 1]
    ])

def mat2quat(m):
    """
    将 3x3 旋转矩阵安全转换为四元数 [x, y, z, w]
    """
    t = np.trace(m[:3, :3])
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        return [(m[2,1] - m[1,2]) / s, (m[0,2] - m[2,0]) / s, (m[1,0] - m[0,1]) / s, 0.25 * s]
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
        return [0.25 * s, (m[0,1] + m[1,0]) / s, (m[0,2] + m[2,0]) / s, (m[2,1] - m[1,2]) / s]
    elif m[1,1] > m[2,2]:
        s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
        return [(m[0,1] + m[1,0]) / s, 0.25 * s, (m[1,2] + m[2,1]) / s, (m[0,2] - m[2,0]) / s]
    else:
        s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
        return [(m[0,2] + m[2,0]) / s, (m[1,2] + m[2,1]) / s, 0.25 * s, (m[1,0] - m[0,1]) / s]

class RosGraspManager(Node):
    def __init__(self):
        super().__init__('ros_grasp_manager')
        self.bridge = CvBridge()
        self.intrinsic_matrix = {'fx': 554.25, 'fy': 554.25, 'cx': 320.0, 'cy': 240.0}
        
        self.rgb_sub = message_filters.Subscriber(self, Image, '/d435/image', qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, '/d435/depth_image', qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.synchronized_callback)
        
        # 🚨 工业级重构 3：订阅关节状态，感知物理限位
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos_profile_sensor_data)
        self.current_joints = {}
        
        self.latest_rgb = None
        self.latest_depth = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.gripper_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.timer = self.create_timer(2.0, self.setup_scene)
        self.get_logger().info("🚀 抓取管理器已就绪，正在同步仿真环境障碍物...")
    def joint_state_cb(self, msg):
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]
    def setup_scene(self):
        self.timer.cancel()
        base_frame = self.get_base_frame()

        # ================= 1. 桌子避障 =================
        table = CollisionObject()
        table.header.frame_id = base_frame
        table.id = "work_table"
        table_box = SolidPrimitive()
        # 尺寸与 spawn_cokes.py 保持一致: TABLE_W=0.6, TABLE_L=1.0, TABLE_H=0.4
        table_box.type, table_box.dimensions = SolidPrimitive.BOX, [0.6, 1.0, 0.4]
        table_pose = Pose()
        table_pose.position.x, table_pose.position.y, table_pose.position.z = 0.6, 0.0, 0.2
        table.primitives.append(table_box)
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD
        self.collision_pub.publish(table)

        # ================= 2. 空心料盒避障 (1个底板 + 4面墙壁) =================
        bin_obj = CollisionObject()
        bin_obj.header.frame_id = base_frame
        bin_obj.id = "collection_bin"

        # 根据 spawn_cokes.py 的参数精准计算世界坐标系下的位置：
        # 盒子中心 X=0.6, Y=0.0。整体中心高度 Z = 0.4(桌高) + 0.15/2(半高) + 0.01(微调) = 0.485
        # 组装参数: [size_x, size_y, size_z], [pos_x, pos_y, pos_z]
        bin_parts = [
            # 底板 (Bottom): 贴着桌面
            ([0.4, 0.6, 0.01], [0.6, 0.0, 0.415]),
            # 前壁 (Front/X+方向)
            ([0.01, 0.6, 0.15], [0.795, 0.0, 0.485]),
            # 后壁 (Back/X-方向)
            ([0.01, 0.6, 0.15], [0.405, 0.0, 0.485]),
            # 左壁 (Left/Y+方向)
            ([0.4, 0.01, 0.15], [0.6, 0.295, 0.485]),
            # 右壁 (Right/Y-方向)
            ([0.4, 0.01, 0.15], [0.6, -0.295, 0.485])
        ]

        for size, pos in bin_parts:
            sp = SolidPrimitive()
            sp.type = SolidPrimitive.BOX
            sp.dimensions = size
            
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pos[0], pos[1], pos[2]
            
            bin_obj.primitives.append(sp)
            bin_obj.primitive_poses.append(pose)
            
        bin_obj.operation = CollisionObject.ADD
        self.collision_pub.publish(bin_obj)

        self.get_logger().info("✅ 避障场景同步完成 (已生成精准的 3D 空心料盒碰撞体！)")

    def synchronized_callback(self, rgb_msg, depth_msg):
        self.latest_rgb = rgb_msg
        self.latest_depth = depth_msg

    def get_base_frame(self):
        for frame in ['base_link', 'base', 'world']:
            try:
                self.tf_buffer.lookup_transform(frame, 'tool0', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
                return frame
            except: continue
        return 'base_link'

    # 新增：补齐缺失的获取当前位姿的方法
    def get_current_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.get_base_frame(), 'tool0', rclpy.time.Time())
            return np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        except Exception:
            return None

    def call_predict_service(self):
        if self.latest_rgb is None: return None
        
        cv_rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, "bgr8")
        if self.latest_depth.encoding == '16UC1':
            cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth, "16UC1")
        else:
            cv_depth = (self.bridge.imgmsg_to_cv2(self.latest_depth, "32FC1") * 1000).astype(np.uint16)
        
        cv2.imwrite(TEMP_RGB, cv_rgb)
        cv2.imwrite(TEMP_DEPTH, cv_depth)

        payload = {
            "rgb_path": TEMP_RGB,
            "depth_path": TEMP_DEPTH,
            **self.intrinsic_matrix,
            "vis_save_path": VIS_PATH
        }
        
        try:
            response = requests.post(SERVICE_URL, json=payload, timeout=10)
            return response.json()
        except Exception as e:
            self.get_logger().error(f"请求 Flask 服务失败: {e}")
            return None

    def draw_and_show_grasps(self, grasps):
        """直接打开由 Flask 服务端渲染好的，带有 3D 姿态箭头的精美图片"""
        if os.path.exists(VIS_PATH):
            os.system(f"xdg-open {VIS_PATH} &")
        else:
            self.get_logger().error(f"❌ 未找到可视化图片: {VIS_PATH}")

    def sync_move(self, target_mat, action_name="移动"):
        # 如果 Action Server 不在线，直接退出防止卡死
        if not self.move_client.server_is_ready():
            self.get_logger().error("❌ 未连接到 /move_action 服务器，请检查 MoveIt 是否正常运行！")
            return False

        tgt_pos = target_mat[:3, 3]
        if tgt_pos[2] < MIN_SAFE_Z:
            self.get_logger().error(f"🛑 目标 Z 高度 ({tgt_pos[2]:.3f}m) 低于安全线 {MIN_SAFE_Z}m，为保护硬件已拦截！")
            return False
        
        
        
        base_frame = self.get_base_frame()
        
        print(f"\n" + "═"*50)
        print(f"⏸️  动作卡点: 准备执行【{action_name}】")
        print(f"🎯  目标坐标 (tool0): X={tgt_pos[0]:.3f}, Y={tgt_pos[1]:.3f}, Z={tgt_pos[2]:.3f}")
        print(f"═"*50)
        user_cmd = input(">>> 按【回车键】发送移动指令，输入 'q' 取消: ")
        if user_cmd.strip().lower() == 'q':
            self.get_logger().warn(f"🛑 用户手动拦截了【{action_name}】动作！")
            return False

        self.get_logger().info(f"▶️ 开始发送 {action_name} 规划请求...")
        
        goal = MoveGroup.Goal()
        goal.request.group_name = ARM_GROUP_NAME
        
        from moveit_msgs.msg import WorkspaceParameters
        ws = WorkspaceParameters()
        ws.header.frame_id = base_frame
        ws.min_corner.x, ws.min_corner.y, ws.min_corner.z = -2.0, -2.0, -2.0
        ws.max_corner.x, ws.max_corner.y, ws.max_corner.z = 2.0, 2.0, 2.0
        goal.request.workspace_parameters = ws
        
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = tgt_pos
        q = mat2quat(target_mat)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        
        c = Constraints()
        p = PositionConstraint()
        p.header.frame_id = base_frame
        p.link_name = "tool0"
        
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type, sp.dimensions = SolidPrimitive.SPHERE, [0.015] 
        bv.primitives.append(sp)
        bv.primitive_poses.append(pose) 
        p.constraint_region = bv
        c.position_constraints.append(p)
        
        o = OrientationConstraint()
        o.header.frame_id = base_frame
        o.link_name = "tool0"
        o.orientation = pose.orientation 
        
        o.absolute_x_axis_tolerance = 0.15 
        o.absolute_y_axis_tolerance = 0.15  
        o.absolute_z_axis_tolerance = 0.15
        o.weight = 1.0
        c.orientation_constraints.append(o)

        goal.request.goal_constraints.append(c)
        goal.request.allowed_planning_time = 8.0
        goal.request.max_velocity_scaling_factor = 0.4
        
        future = self.move_client.send_goal_async(goal)
        
        # 使用更安全的等待方式，配合 MultiThreadedExecutor 解决死锁
        while rclpy.ok() and not future.done(): 
            time.sleep(0.05)
        
        if not future.done() or not future.result().accepted:
            self.get_logger().error(f"❌ {action_name} 规划被 MoveIt 拒绝！(可能是 IK 无解或发生碰撞)")
            return False

        goal_handle = future.result()
        res_future = goal_handle.get_result_async()
        
        while rclpy.ok() and not res_future.done(): 
            time.sleep(0.05)
        # ================= 核心修复：听取 MoveIt 的真实心声 =================
        result_msg = res_future.result().result
        error_code = result_msg.error_code.val
        
        # MoveItErrorCode 中，1 代表 SUCCESS
        if error_code != 1:
            self.get_logger().error(f"❌ MoveIt 动作失败！(错误码: {error_code})")
            # 常见错误码提示：
            # -31: 找不到逆运动学(IK)解 (目标点太远或姿态扭曲)
            # -400: 控制器执行失败 (Gazebo没收到指令)
            return False
        # ================================================================
        curr_p = self.get_current_pose()
        if curr_p is not None:
            dist = np.linalg.norm(curr_p - tgt_pos)
            self.get_logger().info(f"✅ 完成: {action_name} | 偏差: {dist*1000:.1f}mm")
        
        return True

    def execute_grasp_sequence(self, grasp):
        target_base = self.get_base_frame()
        try:
            tf = self.tf_buffer.lookup_transform(target_base, 'camera_depth_optical_frame', rclpy.time.Time())
            T_cam_base = quat2mat([tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w])
            T_cam_base[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        except Exception as e:
            self.get_logger().error(f"TF 变换失败: {e}")
            return

        # 1. 提取相机坐标系下的位置和【真实旋转矩阵】
        cam_pos = np.array(grasp['translation'])
        cam_rot = np.array(grasp['rotation']) 

        # ================= [🚨 核心修复：坐标系映射对齐] =================
        R_graspnet_to_tool0 = np.array([
            [ 0,  0,  1],  
            [ 1,  0,  0],  
            [ 0,  1,  0]   
        ])
       
        # 局部坐标轴重映射 (右乘)
        cam_rot_corrected = cam_rot @ R_graspnet_to_tool0
        # ================================================================

        # 2. 构造完整的相机坐标系抓取位姿
        T_grasp_cam = np.eye(4)
        T_grasp_cam[:3, :3] = cam_rot_corrected
        T_grasp_cam[:3, 3] = cam_pos             
        
        
        T_grasp_base = T_cam_base @ T_grasp_cam
        base_pos = T_grasp_base[:3, 3]
        
        tcp_offset = np.array([0, 0, -TCP_OFFSET_Z])
        tcp_offset_world = T_grasp_base[:3, :3] @ tcp_offset
        final_pos = base_pos + tcp_offset_world
        
        T_final = np.eye(4)
        T_final[:3, :3] = T_grasp_base[:3, :3]
        T_final[:3, 3] = final_pos
        # ==================== 🕵️ 诊断代码区 (请将打印结果发给我分析) ====================
        print("\n" + "▼"*50)
        print("🕵️ 抓取姿态核心诊断日志")
        print("1. [GraspNet 原始输出] (相机坐标系下的旋转矩阵):")
        print(np.round(cam_rot, 3))
        
        print("\n2. [映射修正后] (相机坐标系，已将 X轴下探 转为 Z轴下探):")
        print(np.round(cam_rot_corrected, 3))
        
        print("\n3. [最终目标姿态] (基座坐标系 base_link 下的旋转矩阵):")
        print(np.round(T_final[:3, :3], 3))
        
        # 提取目标姿态的三个坐标轴在世界(基座)坐标系中的真实朝向
        x_axis = T_final[:3, 0] # tool0 的 X 轴向量
        y_axis = T_final[:3, 1] # tool0 的 Y 轴向量 (夹爪闭合方向)
        z_axis = T_final[:3, 2] # tool0 的 Z 轴向量 (伸入/下探方向)
        
        print("\n4. [法兰盘(tool0) 预期朝向向量解析]:")
        print(f"   👇 Z轴 (下探方向): [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]  <-- 正常应接近 [0, 0, -1] 代表向下")
        print(f"   🤏 Y轴 (闭合方向): [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]  <-- 决定手腕偏航角 (转圈圈的原因)")
        print(f"   👉 X轴 (侧面方向): [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]")
        
        # 把刚才用来发给 MoveIt 的四元数也打印出来核对
        q_target = mat2quat(T_final)
        print(f"\n5. [发给 MoveIt 的四元数 (xyzw)]: [{q_target[0]:.4f}, {q_target[1]:.4f}, {q_target[2]:.4f}, {q_target[3]:.4f}]")
        print("▲"*50 + "\n")
        # =========================================================================
        # 6.预抓取位置（叠加局部偏移，沿 Z 轴负方向，增加安全校验）
        T_pre = T_final.copy()
        pre_offset = np.array([0, 0, -PRE_GRASP_OFFSET])
        T_pre[:3, 3] += T_final[:3, :3] @ pre_offset  
        
        if T_pre[2, 3] < MIN_SAFE_Z:
            self.get_logger().error(f"🛑 预抓取 Z 高度 {T_pre[2,3]:.3f} 低于安全线 {MIN_SAFE_Z}")
            return
        
        print("\n🤖 [动作 1/5] 确保夹爪张开...")
        if not self.call_gripper(0.0, "张开夹爪"):
            return # 如果没打开就终止

        success = self.sync_move(T_pre, "预抓取位置 (悬停在物体上方)")
        curr_p = self.get_current_pose()
        deviation = np.linalg.norm(curr_p - T_pre[:3, 3]) if curr_p is not None else 999

        # 2. 智能判断：如果失败，或者虽然成功但偏差 > 20mm (说明被关节限位卡在边缘了)
        if not success or deviation > 0.02:
            self.get_logger().warn(f"⚠️ 发现姿态极度扭曲 (偏差 {deviation*1000:.1f}mm)，触发 180° 翻转自救机制！")
            
            # 构建绕 Z 轴旋转 180 度的矩阵 (Z轴不变，X和Y反向)
            R_z_180 = np.array([
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0,  1]
            ])
            
            # 将最终目标和悬停目标的姿态都翻转 180 度
            T_final[:3, :3] = T_final[:3, :3] @ R_z_180
            T_pre[:3, :3] = T_pre[:3, :3] @ R_z_180

            # 重新尝试悬停
            if not self.sync_move(T_pre, "预抓取位置 (180°翻转后)"):
                self.get_logger().error("❌ 翻转后依然无解，放弃当前抓取点，请重试。")
                return
        # =====================================================================

        # 动作 3: 俯冲深探
        if not self.sync_move(T_final, "步骤 2/3：俯冲到抓取位"): return

        # --- ✨ 新增：到达最后位置后的【偏差计算】 ---
        curr_p = self.get_current_pose()
        if curr_p is not None:
            dist_err = np.linalg.norm(curr_p - T_final[:3, 3])
            print(f"\n📏 [偏差检测] 当前位置与目标偏差: {dist_err*1000:.2f}mm")
            if dist_err > 0.015: # 超过 15mm 提醒
                 print("⚠️ 偏差较大，可能发生碰撞或已触限位！")

        # 动作 4: 用户确认闭合
        cmd = input("\n>>> 观察 Gazebo：位置对了吗？[回车] 闭合，[q] 取消并复位: ")
        if cmd.strip().lower() == 'q':
            self.go_home()
            return

        self.call_gripper(0.8, "步骤 3/3：闭合夹爪")

        # 动作 5: 提起
        T_lift = T_final.copy()
        T_lift[2, 3] += LIFT_HEIGHT
        self.sync_move(T_lift, "提起物体")

        # 动作 6: 放回初始位置
        input("\n>>> [回车] 释放物体并回到 Home 点...")
        self.call_gripper(0.0, "释放物体")
        self.go_home()

    def go_home(self):
        """通过关节空间指令让机械臂安全回到初始位置"""
        self.get_logger().info("🏠 正在请求回到初始 Home 位置...")
        
        # 构造关节目标请求
        goal = MoveGroup.Goal()
        goal.request.group_name = ARM_GROUP_NAME
        
        from moveit_msgs.msg import Constraints, JointConstraint
        constraints = Constraints()
        
        # 定义 6 个关节的名字（必须与你的 UR 机器人一致）
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        # 使用你提供的初始弧度位置
        # HOME_JOINTS = [-0.28, -1.61, 0.91, -0.84, -1.57, -0.24]
        for name, pos in zip(joint_names, HOME_JOINTS):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        
        goal.request.goal_constraints.append(constraints)
        
        # 发送并等待结果（这里使用同步等待逻辑，确保复位完成后再继续下一次扫描）
        future = self.move_client.send_goal_async(goal)
        while rclpy.ok() and not future.done():
            time.sleep(0.1)
        
        if future.result().accepted:
            res_future = future.result().get_result_async()
            while rclpy.ok() and not res_future.done():
                time.sleep(0.1)
            self.get_logger().info("✅ 已经成功回到初始位置。")
        else:
            self.get_logger().error("❌ 回到初始位置的请求被拒绝！")
    def call_gripper(self, pos, action_name="操作夹爪"):
        from control_msgs.msg import GripperCommand as GripperMsg
        self.get_logger().info(f"⏳ 正在请求: {action_name} (位置: {pos})...")
        
        # 1. 确保服务器在线
        if not self.gripper_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("❌ 致命错误: 夹爪控制器 (/robotiq_gripper_controller) 未响应！")
            return False

        # 2. 构造目标
        goal = GripperCommand.Goal()
        goal.command = GripperMsg() 
        goal.command.position = pos
        goal.command.max_effort = 50.0
        
        # 3. 发送目标并等待接受
        future = self.gripper_client.send_goal_async(goal)
        while rclpy.ok() and not future.done(): 
            time.sleep(0.1)
        
        if not future.done() or not future.result().accepted:
            self.get_logger().error(f"❌ 夹爪拒绝了 {action_name} 的请求！")
            return False

        # 4. 严格等待动作物理执行完毕 (🚨 新增了 3 秒超时打破死锁)
        goal_handle = future.result()
        res_future = goal_handle.get_result_async()
        
        start_time = time.time()
        timeout_sec = 4.0  # 设定 4 秒超时
        
        while rclpy.ok() and not res_future.done(): 
            if time.time() - start_time > timeout_sec:
                # 🚨 工业级重构 2：硬件级安全保护！
                self.get_logger().warn(f"⚠️ {action_name} 物理阻塞或超时，立即发送 Cancel 指令保护电机！")
                goal_handle.cancel_goal_async()  # 发送取消指令，切断电流防止死锁烧毁
                time.sleep(0.5) # 给控制器处理 Cancel 的时间
                break  
            time.sleep(0.05) 
            
        self.get_logger().info(f"✅ 完成: {action_name}")
        return True

def main(args=None):
    rclpy.init(args=args)
    node = RosGraspManager()

    def ui_thread(mgr_node):
        mgr_node.move_client.wait_for_server(timeout_sec=5.0)

        while rclpy.ok():
            input("\n>>> [回车] 开启相机扫描...")
            res = mgr_node.call_predict_service()
            
            if res and res.get('status') == 'success':
                objects = res.get('objects', [])
                mgr_node.draw_and_show_grasps(None) # 触发打开图片
                
                # 第一阶段：选择物体
                print(f"\n✨ 识别成功！发现 {len(objects)} 个目标物体:")
                for obj in objects:
                    print(f"  [物体 {obj['obj_id']}] 类别: {obj['class_name']} | 包含 {len(obj['grasps'])} 个优质抓取点")
                
                obj_choice = input(f"\n👉 请输入要抓取的【物体编号】(1-{len(objects)})，或输入 'r' 重扫: ")
                
                if obj_choice.isdigit() and 1 <= int(obj_choice) <= len(objects):
                    selected_obj = objects[int(obj_choice) - 1]
                    grasps = selected_obj['grasps']
                    
                    # 第二阶段：选择该物体上的抓取点
                    # === 修改后的代码 ===
                    # 第二阶段：自动提取成功率（置信度）最高的抓取点
                    print(f"\n🎯 已锁定物体 [{selected_obj['class_name']}]，请参考图片选择抓取点 (图中标号为 {selected_obj['obj_id']}-X):")
                    for i, g in enumerate(grasps):
                        print(f"  [点位 {i+1}] 置信度: {g['score']:.3f} | 深度 Z: {g['translation'][2]:.3f}m | 角度: {g['angle']:.1f}°")
                    
                    grasp_choice = input(f"\n👉 请输入执行的【点位编号】(1-{len(grasps)}): ")
                    if grasp_choice.isdigit() and 1 <= int(grasp_choice) <= len(grasps):
                        idx = int(grasp_choice) - 1
                        mgr_node.execute_grasp_sequence(grasps[idx])
                    else:
                        print("🔄 点位输入无效，已取消执行。")
                else:
                    print("🔄 物体输入无效，已取消执行。")
            else:
                # ================= 🕵️ 诊断弹图 =================
                print(f"\n❌ 识别失败。原因: {res.get('msg', '未知')}")
                print("📸 正在为您弹出相机实时画面，请观察当前视野...")
                mgr_node.draw_and_show_grasps(None) # 失败了也把图片弹出来！
                # ==============================================

    threading.Thread(target=ui_thread, args=(node,), daemon=True).start()
    
    # 修复2：使用 MultiThreadedExecutor，彻底解决由于单线程处理回调导致的 Action 堵塞死锁
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # 修复3：安全的关闭逻辑
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
