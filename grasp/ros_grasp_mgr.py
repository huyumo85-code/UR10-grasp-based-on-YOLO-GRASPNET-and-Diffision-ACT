import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor 
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
from std_msgs.msg import String

# ================= 配置区域 =================
SERVICE_URL = "http://localhost:5005/predict"
ARM_GROUP_NAME = "ur_manipulator"
TCP_OFFSET_Z = 0.112
PRE_GRASP_OFFSET = 0.10
LIFT_HEIGHT = 0.15
MIN_SAFE_Z = 0.48

TEMP_RGB = "/tmp/ros_rgb.png"
TEMP_DEPTH = "/tmp/ros_depth.png"
VIS_PATH = "/tmp/ros_vis.png" 
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
        
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos_profile_sensor_data)
        self.current_joints = {}
        
        self.latest_rgb = None
        self.latest_depth = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.gripper_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.record_pub = self.create_publisher(String, '/record_signal', 10)
        self.timer = self.create_timer(2.0, self.setup_scene)
        self.get_logger().info("🚀 抓取管理器已就绪，正在同步仿真环境障碍物...")

    def joint_state_cb(self, msg):
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]

    def setup_scene(self):
        self.timer.cancel()
        base_frame = self.get_base_frame()

        table = CollisionObject()
        table.header.frame_id = base_frame
        table.id = "work_table"
        table_box = SolidPrimitive()
        table_box.type, table_box.dimensions = SolidPrimitive.BOX, [0.6, 1.0, 0.4]
        table_pose = Pose()
        table_pose.position.x, table_pose.position.y, table_pose.position.z = 0.6, 0.0, 0.2
        table.primitives.append(table_box)
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD
        self.collision_pub.publish(table)

        bin_obj = CollisionObject()
        bin_obj.header.frame_id = base_frame
        bin_obj.id = "collection_bin"

        bin_parts = [
            ([0.4, 0.6, 0.01], [0.6, 0.0, 0.415]),
            ([0.01, 0.6, 0.15], [0.795, 0.0, 0.485]),
            ([0.01, 0.6, 0.15], [0.405, 0.0, 0.485]),
            ([0.4, 0.01, 0.15], [0.6, 0.295, 0.485]),
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
        self.get_logger().info("✅ 避障场景同步完成")

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

    def sync_move(self, target_mat, action_name="移动"):
        if not self.move_client.server_is_ready():
            self.get_logger().error("❌ 未连接到 /move_action 服务器")
            return False

        tgt_pos = target_mat[:3, 3]
        if tgt_pos[2] < MIN_SAFE_Z:
            self.get_logger().error(f"🛑 目标 Z 高度 ({tgt_pos[2]:.3f}m) 低于安全线 {MIN_SAFE_Z}m，已拦截！")
            return False
        
        base_frame = self.get_base_frame()
        self.get_logger().info(f"▶️ 开始发送 {action_name} 规划请求...")
        
        time.sleep(0.1) 
        
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
        
        while rclpy.ok() and not future.done(): 
            time.sleep(0.05)
        
        if not future.done() or not future.result().accepted:
            self.get_logger().error(f"❌ {action_name} 规划被 MoveIt 拒绝！")
            return False

        goal_handle = future.result()
        res_future = goal_handle.get_result_async()
        
        while rclpy.ok() and not res_future.done(): 
            time.sleep(0.05)

        result_msg = res_future.result().result
        error_code = result_msg.error_code.val
        
        if error_code != 1:
            self.get_logger().error(f"❌ MoveIt 动作失败！(错误码: {error_code})")
            return False

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

        cam_pos = np.array(grasp['translation'])
        cam_rot = np.array(grasp['rotation']) 

        R_graspnet_to_tool0 = np.array([
            [ 0,  0,  1],  
            [ 1,  0,  0],  
            [ 0,  1,  0]   
        ])
        cam_rot_corrected = cam_rot @ R_graspnet_to_tool0

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

        T_pre = T_final.copy()
        pre_offset = np.array([0, 0, -PRE_GRASP_OFFSET])
        T_pre[:3, 3] += T_final[:3, :3] @ pre_offset  
        
        if T_pre[2, 3] < MIN_SAFE_Z:
            self.get_logger().error(f"🛑 预抓取 Z 高度 {T_pre[2,3]:.3f} 低于安全线 {MIN_SAFE_Z}")
            print("❌ [RECORD_ABORT] 目标非法，当前数据作废！")
            self.record_pub.publish(String(data='ABORT'))
            return
        
        if not self.call_gripper(0.0, "张开夹爪"):
            print("❌ [RECORD_ABORT] 夹爪未能打开，当前数据作废！")
            self.record_pub.publish(String(data='ABORT'))
            return

        # 🌟🌟🌟 录制起点挪回这里：开始记录长途飞行！🌟🌟🌟
        print("\n" + "="*50)
        print("🟢 [RECORD_START] 开始录制全过程（包含长途飞行）！")
        self.record_pub.publish(String(data='START'))
        print("="*50 + "\n")

        # 强制给 0.5 秒静止缓冲，让模型看清楚“起飞前”的全局画面
        time.sleep(0.5)

        # 1. 飞行到悬停位置 (这段优雅的空中曲线将被全程录下！)
        success = self.sync_move(T_pre, "预抓取位置 (悬停在物体上方)")
        if not success:
            print("❌ [RECORD_ABORT] 悬停失败，废弃当前数据！")
            self.record_pub.publish(String(data='ABORT'))
            return
            
        curr_p = self.get_current_pose()
        deviation = np.linalg.norm(curr_p - T_pre[:3, 3]) if curr_p is not None else 999

        if deviation > 0.02:
            self.get_logger().warn(f"⚠️ 触发 180° 翻转自救...")
            R_z_180 = np.array([
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0,  1]
            ])
            T_final[:3, :3] = T_final[:3, :3] @ R_z_180
            T_pre[:3, :3] = T_pre[:3, :3] @ R_z_180

            if not self.sync_move(T_pre, "预抓取位置 (180°翻转后)"):
                print("❌ [RECORD_ABORT] 翻转后依然无解，废弃当前数据！")
                self.record_pub.publish(String(data='ABORT'))
                return

        # 2. 开始俯冲
        if not self.sync_move(T_final, "步骤 2/3：俯冲到抓取位"): 
            print("❌ [RECORD_ABORT] 俯冲失败，废弃当前数据！")
            self.record_pub.publish(String(data='ABORT'))
            return

        print("\n⏳ 俯冲到位，物理缓冲 0.5 秒消除震动...")
        time.sleep(0.5) 
        
        # ... 后面的闭合、抬升、结束信标保持不变 ...

        self.call_gripper(0.8, "步骤 3/3：闭合夹爪")
        print("⏳ 等待夹爪建立稳定摩擦力 1.0 秒...")
        time.sleep(1.0) 

        T_lift = T_final.copy()
        T_lift[2, 3] += LIFT_HEIGHT
        if not self.sync_move(T_lift, "提起物体"):
            print("❌ [RECORD_ABORT] 抬升失败，废弃当前数据！")
            self.record_pub.publish(String(data='ABORT'))
            return

        # ================= 🎥 录制结束信标 =================
        print("\n" + "="*50)
        print("🔴 [RECORD_END] 抬升完成！请保存当前段落。")
        self.record_pub.publish(String(data='END'))
        print("="*50 + "\n")

        print("♻️ 准备释放物体并回城 (此段动作请勿录制进数据集)...")
        time.sleep(1.0)
        self.call_gripper(0.0, "释放物体")
        self.go_home()

    def go_home(self):
        self.get_logger().info("🏠 正在请求回到初始 Home 位置...")
        goal = MoveGroup.Goal()
        goal.request.group_name = ARM_GROUP_NAME
        
        from moveit_msgs.msg import Constraints, JointConstraint
        constraints = Constraints()
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        for name, pos in zip(joint_names, HOME_JOINTS):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        
        goal.request.goal_constraints.append(constraints)
        
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
        
        if not self.gripper_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("❌ 致命错误: 夹爪控制器未响应！")
            return False

        goal = GripperCommand.Goal()
        goal.command = GripperMsg() 
        goal.command.position = pos
        goal.command.max_effort = 50.0
        
        future = self.gripper_client.send_goal_async(goal)
        while rclpy.ok() and not future.done(): 
            time.sleep(0.1)
        
        if not future.done() or not future.result().accepted:
            self.get_logger().error(f"❌ 夹爪拒绝了 {action_name} 的请求！")
            return False

        goal_handle = future.result()
        res_future = goal_handle.get_result_async()
        
        start_time = time.time()
        timeout_sec = 4.0  
        
        while rclpy.ok() and not res_future.done(): 
            if time.time() - start_time > timeout_sec:
                self.get_logger().warn(f"⚠️ {action_name} 物理阻塞或超时，发送 Cancel！")
                goal_handle.cancel_goal_async()  
                time.sleep(0.5) 
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
            input("\n>>> [摆好易拉罐后，按回车] 开启自动识别与录制流程...")
            res = mgr_node.call_predict_service()
            
            if res and res.get('status') == 'success':
                objects = res.get('objects', [])
                if not objects:
                    print("❌ 未识别到任何物体，请调整易拉罐位置。")
                    continue
                
                selected_obj = objects[0]
                grasps = selected_obj['grasps']

                depth_rank = selected_obj.get('depth_rank', 1)
                avg_depth = selected_obj.get('avg_depth_m', -1.0)
                total = len(objects)
                print(f"\n✨ 识别成功！共检测到 {total} 个物体，按深度优先级自动排序：")
                for obj in objects:
                    tag = " <-- 当前目标" if obj is selected_obj else ""
                    print(f"   #{obj.get('depth_rank','-')} {obj['class_name']} | 深度: {obj.get('avg_depth_m',0):.3f}m{tag}")
                print(f"🎯 锁定第 {depth_rank} 优先目标: {selected_obj['class_name']} | 深度: {avg_depth:.3f}m | 置信度: {grasps[0]['score']:.3f}")
                
                mgr_node.execute_grasp_sequence(grasps[0])
                
            else:
                print(f"\n❌ 识别失败。原因: {res.get('msg', '未知')}")

    threading.Thread(target=ui_thread, args=(node,), daemon=True).start()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
