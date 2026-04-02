import sys
import rclpy
from rclpy.node import Node
import numpy as np
import requests
import threading
import time
import cv2
import json
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand

import tf2_ros
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, WorkspaceParameters
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

# 你录制时使用的绝对起点
HOME_JOINTS = [-0.28, -1.61, 0.91, -0.84, -1.57, -0.24]

# 相机内参（与 grasp_service 保持一致）
CAMERA_FX, CAMERA_FY = 554.25, 554.25
CAMERA_CX, CAMERA_CY = 320.0, 240.0
# 粗定位：MoveIt 悬停高度（目标质心正上方）
PRE_GRASP_HEIGHT = 0.15

class TemporalEnsembler:
    def __init__(self, horizon=16):
        self.horizon = horizon
        self.preds = []
        self.step = 0

    def update_and_get(self, new_traj, lookahead=2):
        self.preds.append((self.step, np.array(new_traj)))
        self.preds = [p for p in self.preds if p[0] + self.horizon > self.step + lookahead]

        current_actions = []
        weights = []
        for p_step, traj in self.preds:
            idx = (self.step + lookahead) - p_step
            if 0 <= idx < self.horizon:
                current_actions.append(traj[idx])
                weights.append(np.exp(-idx * 0.5))

        # 修复：空列表 fallback，防止除零崩溃
        if not current_actions:
            self.step += 1
            return np.array(new_traj[0])

        weights = np.array(weights) / np.sum(weights)
        ensembled_action = np.average(current_actions, axis=0, weights=weights)
        self.step += 1
        return ensembled_action

class RosDiffusionManager(Node):
    def __init__(self):
        super().__init__(
            'ros_act_manager',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_global_rgb = None
        self.vis_img = None
        self.current_joint_pos = None
        self.is_gripping = False      # 夹爪状态锁（一旦闭合就锁死）
        self.last_sent_grip_cmd = None

        self.arm_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.gripper_joint_name = "robotiq_85_left_knuckle_joint"

        self.latest_depth = None

        self.create_subscription(Image, '/d435/image', self.img_cb, 10)
        self.create_subscription(Image, '/d435/depth_image', self.depth_cb, 10)
        self.create_subscription(Image, '/camera_global/image', self.global_img_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        self.joint_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

        self._gripper_action_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self._log_counter = 0
        self._act_counter = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.move_client = ActionClient(self, MoveGroup, '/move_action')

    def send_gripper_goal(self, position):
        if self.last_sent_grip_cmd == position: return
        self.last_sent_grip_cmd = position
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 100.0
        if not self._gripper_action_client.wait_for_server(timeout_sec=0.1): return
        self._gripper_action_client.send_goal_async(goal_msg)

    def img_cb(self, m): self.latest_rgb = m
    def depth_cb(self, m): self.latest_depth = m
    def global_img_cb(self, m): self.latest_global_rgb = m

    def joint_cb(self, msg):
        pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
        try:
            temp_pos = [pos_dict[name] for name in self.arm_joint_names]
            temp_pos.append(pos_dict.get(self.gripper_joint_name, 0.0))
            self.current_joint_pos = temp_pos
            self._log_counter = getattr(self, '_log_counter', 0) + 1
        except KeyError: pass

    def get_tcp_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        except: return None

    def go_home_direct(self):
        print("🏠 正在强制复位到起跑线 (Home)...")
        self.is_gripping = False
        self.last_sent_grip_cmd = None
        self.send_gripper_goal(0.0)

        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joint_names
        arm_p = JointTrajectoryPoint()
        arm_p.positions = HOME_JOINTS
        arm_p.time_from_start.sec = 3
        arm_msg.points.append(arm_p)
        self.joint_pub.publish(arm_msg)
        time.sleep(3.5)
        print("✅ 起跑线就绪！")

    def publish_action(self, action):
        arm_msg = JointTrajectory()
        arm_msg.header.stamp.sec = 0
        arm_msg.header.stamp.nanosec = 0
        arm_msg.joint_names = self.arm_joint_names

        arm_p = JointTrajectoryPoint()

        if self.current_joint_pos is not None:
            current_arm = np.array(self.current_joint_pos[:6])
            target_arm = np.array(action[:6])
            if getattr(self, '_act_counter', 0) % 10 == 0:
                targets = [round(x, 2) for x in target_arm.tolist()]
                print(f"🎯 Step {self._act_counter:3d} | Target Joints: {targets}")
            self._act_counter = getattr(self, '_act_counter', 0) + 1
            # 统一获取 TCP 高度，用于 delta 截断和夹爪 z 门控
            tcp = self.get_tcp_pose()
            tcp_z = tcp[2] if tcp is not None else getattr(self, '_last_tcp_z', 1.0)
            self._last_tcp_z = tcp_z

            # 夹爪一旦锁死，手臂冻结在当前位置，不再发新目标
            if self.is_gripping:
                arm_p.positions = [float(x) for x in current_arm]
            else:
                diff = target_arm - current_arm
                max_allowed = 0.03 if tcp_z < 0.65 else 0.05
                # 指尖 = tool0 - 0.112m，盒顶 z=0.55，故 tool0 < 0.662 时指尖已入盒
                pan_limit = 0.005 if tcp_z < 0.67 else max_allowed
                per_joint_limits = np.array([pan_limit, max_allowed, max_allowed,
                                             max_allowed, max_allowed, max_allowed])
                clipped_diff = np.clip(diff, -per_joint_limits, per_joint_limits)
                arm_p.positions = [float(x) for x in (current_arm + clipped_diff)]
        else:
            arm_p.positions = [float(x) for x in action[:6]]

        # 绝对位置安全边界：防止模型累积漂移到离 HOME 过远的位置
        _home = np.array(HOME_JOINTS)
        _margin = np.array([0.35, 0.5, 0.6, 0.7, 0.6, 1.5])
        safe_pos = np.clip(np.array(arm_p.positions), _home - _margin, _home + _margin)
        arm_p.positions = [float(x) for x in safe_pos]

        arm_p.velocities = []
        arm_p.accelerations = []
        arm_p.time_from_start.nanosec = 200_000_000
        arm_msg.points.append(arm_p)
        self.joint_pub.publish(arm_msg)

        # 夹爪门控：阈值提高到 0.25，且 z < 0.62m 时才允许锁定（确保臂已到罐子上方）
        target_grip = float(action[6])
        # 指尖到罐子中心 z=0.46，tool0 需到 0.46+0.112=0.572
        if target_grip > 0.25 and getattr(self, '_last_tcp_z', 1.0) < 0.58:
            self.is_gripping = True
        if self.is_gripping:
            self.send_gripper_goal(0.8)
        else:
            self.send_gripper_goal(0.0)

    # ==============================================================
    # 🌟 第一阶段：利用深度图计算目标3D质心（相机坐标 → base_link）
    # ==============================================================
    def get_target_3d_base(self, polygon):
        """
        给定多边形像素坐标，利用深度图反投影到相机坐标系，
        再通过 TF 变换到 base_link，返回 (pos_base_3d, avg_depth_m) 或 None。
        """
        if self.latest_depth is None:
            return None
        try:
            if self.latest_depth.encoding == '16UC1':
                cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth, "16UC1")
            else:
                cv_depth = (self.bridge.imgmsg_to_cv2(self.latest_depth, "32FC1") * 1000).astype(np.uint16)

            h, w = cv_depth.shape
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(obj_mask, [pts], 255)

            depth_vals = cv_depth[obj_mask > 0]
            valid_depth = depth_vals[depth_vals > 0]
            if len(valid_depth) == 0:
                return None

            depth_mm = float(np.median(valid_depth))
            z_cam = depth_mm / 1000.0

            poly_arr = np.array(polygon, dtype=np.float32)
            u = float(np.mean(poly_arr[:, 0]))
            v = float(np.mean(poly_arr[:, 1]))

            x_cam = (u - CAMERA_CX) * z_cam / CAMERA_FX
            y_cam = (v - CAMERA_CY) * z_cam / CAMERA_FY
            pos_cam_h = np.array([x_cam, y_cam, z_cam, 1.0])

            tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_depth_optical_frame',
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            q = tf.transform.rotation
            t = tf.transform.translation
            x, y, z, w_q = q.x, q.y, q.z, q.w
            T = np.eye(4)
            T[:3, :3] = np.array([
                [1-2*y*y-2*z*z,   2*x*y-2*z*w_q, 2*x*z+2*y*w_q],
                [2*x*y+2*z*w_q,   1-2*x*x-2*z*z, 2*y*z-2*x*w_q],
                [2*x*z-2*y*w_q,   2*y*z+2*x*w_q, 1-2*x*x-2*y*y]
            ])
            T[:3, 3] = [t.x, t.y, t.z]

            pos_base = (T @ pos_cam_h)[:3]
            return pos_base, z_cam

        except Exception as e:
            self.get_logger().warn(f"3D目标定位失败: {e}")
            return None

    # ==============================================================
    # 🌟 第一阶段：MoveIt 粗定位 — 悬停至目标正上方
    # ==============================================================
    def moveit_approach(self, target_3d_base, pre_height=PRE_GRASP_HEIGHT):
        """
        使用 MoveIt 将 TCP 移动到目标质心正上方 pre_height 处。
        保持当前末端姿态不变，只改变位置。
        """
        if not self.move_client.wait_for_server(timeout_sec=3.0):
            print("❌ MoveIt 服务器未响应，粗定位失败")
            return False

        pre_pos = target_3d_base.copy()
        pre_pos[2] += pre_height

        print(f"▶️  粗定位目标：[{pre_pos[0]:.3f}, {pre_pos[1]:.3f}, {pre_pos[2]:.3f}] (目标上方 {pre_height*100:.0f}cm)")

        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            current_ori = t.transform.rotation
        except Exception as e:
            print(f"❌ 无法获取当前末端姿态: {e}")
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_manipulator"

        ws = WorkspaceParameters()
        ws.header.frame_id = "base_link"
        ws.min_corner.x, ws.min_corner.y, ws.min_corner.z = -2.0, -2.0, -2.0
        ws.max_corner.x, ws.max_corner.y, ws.max_corner.z = 2.0, 2.0, 2.0
        goal.request.workspace_parameters = ws

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(pre_pos[0]), float(pre_pos[1]), float(pre_pos[2])
        pose.orientation = current_ori

        c = Constraints()
        p = PositionConstraint()
        p.header.frame_id = "base_link"
        p.link_name = "tool0"
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type, sp.dimensions = SolidPrimitive.SPHERE, [0.03]
        bv.primitives.append(sp)
        bv.primitive_poses.append(pose)
        p.constraint_region = bv
        c.position_constraints.append(p)

        o = OrientationConstraint()
        o.header.frame_id = "base_link"
        o.link_name = "tool0"
        o.orientation = current_ori
        o.absolute_x_axis_tolerance = 0.3
        o.absolute_y_axis_tolerance = 0.3
        o.absolute_z_axis_tolerance = 0.3
        o.weight = 1.0
        c.orientation_constraints.append(o)

        goal.request.goal_constraints.append(c)
        goal.request.allowed_planning_time = 8.0
        goal.request.max_velocity_scaling_factor = 0.4

        future = self.move_client.send_goal_async(goal)
        while rclpy.ok() and not future.done(): time.sleep(0.05)

        if not future.result().accepted:
            print("❌ 粗定位规划被 MoveIt 拒绝（超出工作空间或自碰撞）")
            return False

        res_future = future.result().get_result_async()
        while rclpy.ok() and not res_future.done(): time.sleep(0.05)

        err_code = res_future.result().result.error_code.val
        if err_code != 1:
            print(f"❌ 粗定位执行失败 (错误码: {err_code})")
            return False

        print("✅ 粗定位完成！手眼相机已对准目标上方，准备切入 ACT 精抓")
        return True

    def smart_extrapolate_grasp(self, p_past, p_curr, ext_dist=0.08, lift_dist=0.15):
        print("\n⚠️ 触发【智能轨迹外推】接管控制权！")
        if not self.move_client.wait_for_server(timeout_sec=2.0):
            print("❌ 找不到 MoveIt 服务器，外挂失效！")
            return

        try:
            t = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            current_ori = t.transform.rotation

            if p_past is None or p_curr is None:
                vec = np.array([0.0, 0.0, -1.0])
            else:
                vec = p_curr - p_past
                norm = np.linalg.norm(vec)
                if norm < 0.01:
                    vec = np.array([0.0, 0.0, -1.0])
                else:
                    vec = vec / norm
                    if vec[2] > -0.1:
                        vec[2] = -0.5
                        vec = vec / np.linalg.norm(vec)

            target_down = p_curr + vec * ext_dist
            print(f"📐 AI 攻击向量: [{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}]")
            print(f"🎯 延长终点: [{target_down[0]:.3f}, {target_down[1]:.3f}, {target_down[2]:.3f}]")

            def move_cartesian_to(target_xyz, action_name):
                print(f"👉 规划中: {action_name}...")
                goal = MoveGroup.Goal()
                goal.request.group_name = "ur_manipulator"

                ws = WorkspaceParameters()
                ws.header.frame_id = "base_link"
                ws.min_corner.x, ws.min_corner.y, ws.min_corner.z = -2.0, -2.0, -2.0
                ws.max_corner.x, ws.max_corner.y, ws.max_corner.z = 2.0, 2.0, 2.0
                goal.request.workspace_parameters = ws

                pose = Pose()
                pose.position.x = float(target_xyz[0])
                pose.position.y = float(target_xyz[1])
                pose.position.z = float(target_xyz[2])
                pose.orientation = current_ori

                c = Constraints()
                p = PositionConstraint()
                p.header.frame_id = "base_link"
                p.link_name = "tool0"
                bv = BoundingVolume()
                sp = SolidPrimitive()
                sp.type, sp.dimensions = SolidPrimitive.SPHERE, [0.025]
                bv.primitives.append(sp)
                bv.primitive_poses.append(pose)
                p.constraint_region = bv
                c.position_constraints.append(p)

                o = OrientationConstraint()
                o.header.frame_id = "base_link"
                o.link_name = "tool0"
                o.orientation = pose.orientation
                o.absolute_x_axis_tolerance = 0.25
                o.absolute_y_axis_tolerance = 0.25
                o.absolute_z_axis_tolerance = 0.25
                o.weight = 1.0
                c.orientation_constraints.append(o)

                goal.request.goal_constraints.append(c)
                goal.request.allowed_planning_time = 5.0
                goal.request.max_velocity_scaling_factor = 0.1

                future = self.move_client.send_goal_async(goal)
                while rclpy.ok() and not future.done(): time.sleep(0.05)

                if not future.result().accepted:
                    print(f"❌ {action_name} 规划被拒绝！")
                    return False

                res_future = future.result().get_result_async()
                while rclpy.ok() and not res_future.done(): time.sleep(0.05)

                err_code = res_future.result().result.error_code.val
                if err_code != 1:
                    print(f"❌ {action_name} 执行失败！错误码: {err_code}")
                    return False

                print(f"✅ {action_name} 完成！")
                return True

            if move_cartesian_to(target_down, f"垂直下探 {ext_dist*100:.0f} cm"):
                print("✊ 到达终点，强行闭合夹爪！")
                self.send_gripper_goal(0.8)
                time.sleep(1.0)
                target_up = target_down + np.array([0, 0, lift_dist])
                move_cartesian_to(target_up, f"抓取后垂直抬升 {lift_dist*100:.0f} cm")
                print("🎉 智能轨迹外挂完成！")
            else:
                print("⚠️ 下潜失败，已安全截断！")

        except Exception as e:
            print(f"❌ 轨迹延伸异常: {e}")

    def show_vision_window(self, cv_img, objects):
        vis_img = cv_img.copy()
        for obj in objects:
            poly = np.array(obj['polygon'], dtype=np.int32)
            rank = obj.get('depth_rank', obj.get('obj_id', '?'))
            depth_m = obj.get('avg_depth_m', 0.0)
            color = (0, 255, 0) if rank == 1 else (255, 180, 0)
            cv2.polylines(vis_img, [poly], True, color, 2)
            label = f"#{rank} {obj['class_name']} ({depth_m:.2f}m)"
            top_pt = tuple(poly[poly[:, 1].argmin()])
            cv2.putText(vis_img, label, (top_pt[0], max(top_pt[1]-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        self.vis_img = vis_img

def ui_logic_thread(node):
    while rclpy.ok():
        input("\n>>> [按回车] 启动两阶段精确抓取 (粗定位 → ACT 精抓)...")

        # 等待所有传感器就绪
        for _ in range(30):
            if node.latest_rgb is not None and node.latest_global_rgb is not None and node.latest_depth is not None:
                break
            time.sleep(0.2)
        else:
            print("❌ 错误：相机数据未就绪 (RGB / Depth / Global 至少一路缺失)")
            continue

        node.go_home_direct()

        try:
            # ============================================================
            # 【第一阶段 · 感知】YOLO 检测 + 深度反投影 + 优先级排序
            # ============================================================
            cv_img = node.bridge.imgmsg_to_cv2(node.latest_rgb, "bgr8")
            _, encoded = cv2.imencode('.png', cv_img)
            resp = requests.post("http://127.0.0.1:5005/detect_objects",
                                 files={'image_hand': encoded.tobytes()}, timeout=5)

            if resp.status_code != 200 or resp.json().get('status') != 'success':
                print("❌ 识别服务响应异常")
                continue

            raw_objects = resp.json().get('objects', [])
            if not raw_objects:
                print("❌ 未识别到任何物体")
                continue

            # 计算每个物体的 3D 质心
            objects_3d = []
            for obj in raw_objects:
                result = node.get_target_3d_base(obj['polygon'])
                if result is not None:
                    pos_base, depth_m = result
                    obj['pos_3d'] = pos_base.tolist()
                    obj['avg_depth_m'] = depth_m
                    objects_3d.append(obj)

            if not objects_3d:
                print("❌ 无法获取任何物体的 3D 位置（深度图无有效数据？）")
                continue

            # 按相机深度升序：最近 = 最顶层 = 最高优先级
            objects_3d.sort(key=lambda o: o['avg_depth_m'])
            for i, obj in enumerate(objects_3d):
                obj['depth_rank'] = i + 1

            print(f"\n✅ 检测到 {len(objects_3d)} 个目标，按深度优先级排序：")
            for obj in objects_3d:
                pos = obj['pos_3d']
                print(f"   #{obj['depth_rank']} {obj['class_name']:10s} | "
                      f"深度: {obj['avg_depth_m']:.3f}m | "
                      f"世界坐标: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

            node.show_vision_window(cv_img, objects_3d)

            # 用户选择目标（默认 #1 最顶层）
            choice = input(
                f"\n👉 默认抓取 #{objects_3d[0]['depth_rank']} ({objects_3d[0]['class_name']})，"
                f"回车确认 / 输入编号选择 / q 退出: "
            ).strip()

            if choice.lower() == 'q':
                node.vis_img = None
                continue

            if choice.isdigit():
                rank = int(choice)
                matched = [o for o in objects_3d if o['depth_rank'] == rank]
                target_obj = matched[0] if matched else objects_3d[0]
            else:
                target_obj = objects_3d[0]

            node.vis_img = None
            target_pos = np.array(target_obj['pos_3d'])
            print(f"\n🎯 锁定目标: #{target_obj['depth_rank']} {target_obj['class_name']} "
                  f"@ [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

            # ============================================================
            # 【第一阶段 · 粗定位】MoveIt 移至目标正上方
            # ============================================================
            print(f"\n[第一阶段] MoveIt 粗定位 → 移至目标上方 {PRE_GRASP_HEIGHT*100:.0f}cm ...")
            if not node.moveit_approach(target_pos):
                print("❌ 粗定位失败，终止本次任务")
                continue

            time.sleep(0.5)   # 等震动消散，画面稳定

            # ============================================================
            # 【第二阶段】ACT 精细抓取（从已靠近目标的位置出发）
            # ============================================================
            print(f"\n[第二阶段] ACT 精细抓取开始 ...")
            requests.post("http://127.0.0.1:5006/reset")

            node.is_gripping = False
            node.last_sent_grip_cmd = None
            node._act_counter = 0
            pose_history = {}
            grip_debounce = 0
            stall_count = 0
            grip_success = False

            for i in range(200):   # 已靠近目标，200 步足够
                loop_start = time.time()
                if not rclpy.ok(): break

                # 悬停检测（step>=30 后开始监控，粗定位后距目标更近，门限提前）
                if i % 10 == 0:
                    p = node.get_tcp_pose()
                    if p is not None:
                        pose_history[i] = p
                        if i >= 30 and (i - 20) in pose_history:
                            dist = np.linalg.norm(p - pose_history[i - 20])
                            stall_count = stall_count + 1 if dist < 0.003 else 0
                        if stall_count >= 3:
                            print(f"🔍 [step {i}] 检测到手臂持续悬停，触发强制垂直俯冲！")
                            break

                img_h = node.bridge.imgmsg_to_cv2(node.latest_rgb, "bgr8")
                img_g = node.bridge.imgmsg_to_cv2(node.latest_global_rgb, "bgr8")
                _, enc_h = cv2.imencode('.png', img_h)
                _, enc_g = cv2.imencode('.png', img_g)
                current_state_data = json.dumps(node.current_joint_pos)

                act_resp = requests.post(
                    "http://127.0.0.1:5006/get_action",
                    files={'image_hand': enc_h.tobytes(), 'image_global': enc_g.tobytes()},
                    data={'current_state': current_state_data})

                if act_resp.status_code == 200:
                    action_step = act_resp.json().get('action')
                    if action_step:
                        node.publish_action(action_step)

                        # 防抖：连续5步输出 > 0.25 且已运行 ≥30 步才确认夹住
                        if i >= 30 and action_step[6] > 0.25:
                            grip_debounce += 1
                        else:
                            grip_debounce = 0

                        if i % 10 == 0:
                            p = node.get_tcp_pose()
                            tcp_z = p[2] if p is not None else 0.0
                            lock_status = "🔒 已锁死" if node.is_gripping else f"等待(防抖:{grip_debounce}/5)"
                            print(f"  [精抓 {i:03d}/200] Grip: {action_step[6]:.3f} | z={tcp_z:.4f} | {lock_status}")

                        if grip_debounce >= 5:
                            print("🎉 模型持续闭合夹爪 5 步，确认触发，跳出循环！")
                            grip_success = True
                            break
                    else:
                        break
                else:
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, 0.1 - elapsed))

            # 等待控制器执行完上一条指令，避免与 MoveIt 冲突
            time.sleep(0.8)

            p_curr = node.get_tcp_pose()
            print(f"🎯 当前位置: {p_curr}")

            if grip_success:
                print("🎉 夹爪确认夹住，保持闭合，自适应对准罐顶...")
                node.send_gripper_goal(0.8)
                time.sleep(0.3)
                ext_dist = max(0.02, p_curr[2] - 0.572) if p_curr is not None else 0.10
                print(f"  自适应俯冲距离: {ext_dist*100:.1f} cm (当前z={p_curr[2]:.3f})")
                node.smart_extrapolate_grasp(None, p_curr, ext_dist=ext_dist, lift_dist=0.15)
            else:
                print("⚠️ 未确认夹住，张爪俯冲兜底...")
                node.send_gripper_goal(0.0)
                time.sleep(0.5)
                node.smart_extrapolate_grasp(None, p_curr, ext_dist=0.12, lift_dist=0.15)

            print("✅ 两阶段 ACT 任务圆满结束")
            node.go_home_direct()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ 系统错误: {e}")

def main():
    rclpy.init(args=sys.argv)
    node = RosDiffusionManager()
    t = threading.Thread(target=ui_logic_thread, args=(node,), daemon=True)
    t.start()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.vis_img is not None:
                cv2.imshow("Detection_Debug", node.vis_img)
                cv2.waitKey(1)
            else:
                try:
                    if cv2.getWindowProperty("Detection_Debug", cv2.WND_PROP_VISIBLE) > 0:
                        cv2.destroyWindow("Detection_Debug")
                except: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
