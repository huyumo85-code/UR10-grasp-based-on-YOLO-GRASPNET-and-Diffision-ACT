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

        self.create_subscription(Image, '/d435/image', self.img_cb, 10)
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
                clipped_diff = np.clip(diff, -max_allowed, max_allowed)
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
        if target_grip > 0.25 and getattr(self, '_last_tcp_z', 1.0) < 0.62:
            self.is_gripping = True
        if self.is_gripping:
            self.send_gripper_goal(0.8)
        else:
            self.send_gripper_goal(0.0)

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
            cv2.polylines(vis_img, [poly], True, (0, 255, 0), 2)
            cv2.putText(vis_img, f"ID:{obj['obj_id']} {obj['class_name']}",
                        (poly[0][0], poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.vis_img = vis_img

def ui_logic_thread(node):
    while rclpy.ok():
        input("\n>>> [按回车] UR10 ACT 端到端抓取 (act)...")
        if node.latest_rgb is None or node.latest_global_rgb is None:
            print("❌ 错误：相机数据未就绪")
            continue

        node.go_home_direct()

        try:
            cv_img = node.bridge.imgmsg_to_cv2(node.latest_rgb, "bgr8")
            _, encoded = cv2.imencode('.png', cv_img)
            resp = requests.post("http://127.0.0.1:5005/detect_objects", files={'image_hand': encoded.tobytes()})

            if resp.status_code == 200 and resp.json()['status'] == 'success':
                res = resp.json()
                if not res['objects']:
                    print("❌ 未识别到任何物体")
                    continue

                node.show_vision_window(cv_img, res['objects'])
                choice = input("👉 确认开始执行 (按回车继续，q 退出): ")
                if choice.lower() == 'q': continue

                node.vis_img = None
                print(f"🚀 ACT 轨迹执行中...")

                # 清空 LeRobot 的内部记忆队列
                requests.post("http://127.0.0.1:5006/reset")

                node.is_gripping = False
                node.last_sent_grip_cmd = None
                node._act_counter = 0
                pose_history = {}
                grip_debounce = 0   # 防抖：连续5步超阈值且 i>=30 才退出
                stall_count = 0     # 悬停计数
                stall_triggered = False
                grip_success = False

                for i in range(300):
                    loop_start = time.time()
                    if not rclpy.ok(): break

                    # 悬停检测：每10步采一次 TCP 位置
                    if i % 10 == 0:
                        p = node.get_tcp_pose()
                        if p is not None:
                            pose_history[i] = p
                            if i >= 50 and (i - 20) in pose_history:
                                dist = np.linalg.norm(p - pose_history[i - 20])
                                if dist < 0.003:   # 20步内位移 < 3mm
                                    stall_count += 1
                                else:
                                    stall_count = 0
                            if stall_count >= 3:
                                print(f"🔍 [step {i}] 检测到手臂持续悬停，触发强制垂直俯冲！")
                                stall_triggered = True
                                break

                    img_h = node.bridge.imgmsg_to_cv2(node.latest_rgb, "bgr8")
                    img_g = node.bridge.imgmsg_to_cv2(node.latest_global_rgb, "bgr8")
                    _, enc_h = cv2.imencode('.png', img_h)
                    _, enc_g = cv2.imencode('.png', img_g)
                    current_state_data = json.dumps(node.current_joint_pos)

                    act_resp = requests.post("http://127.0.0.1:5006/get_action",
                                    files={'image_hand': enc_h.tobytes(), 'image_global': enc_g.tobytes()},
                                    data={'current_state': current_state_data})

                    if act_resp.status_code == 200:
                        action_step = act_resp.json().get('action')
                        if action_step:
                            node.publish_action(action_step)

                            # grip_debounce：连续5步模型原始输出 > 0.25 且至少运行30步才触发
                            if i >= 30 and action_step[6] > 0.25:
                                grip_debounce += 1
                            else:
                                grip_debounce = 0

                            if i % 10 == 0:
                                p = node.get_tcp_pose()
                                tcp_z = p[2] if p is not None else 0.0
                                lock_status = "🔒 已锁死" if node.is_gripping else f"等待(防抖:{grip_debounce}/5)"
                                print(f"  Progress: {i:03d}/300 | Grip: {action_step[6]:.3f} | z={tcp_z:.4f} | {lock_status}")

                            if grip_debounce >= 5:
                                print(f"🎉 模型持续闭合夹爪 5 步，确认触发，跳出循环！")
                                grip_success = True
                                break
                        else: break
                    else: break

                    elapsed = time.time() - loop_start
                    time.sleep(max(0.0, 0.1 - elapsed))

                # 等待控制器执行完上一条指令，避免与 MoveIt 冲突
                time.sleep(0.8)

                p_curr = node.get_tcp_pose()
                print(f"🎯 当前位置: {p_curr}")

                if grip_success:
                    # 模型已确认夹住，保持夹爪闭合，自适应俯冲对准罐子顶端后抬升
                    print("🎉 夹爪确认夹住，保持闭合，自适应对准罐顶...")
                    node.send_gripper_goal(0.8)
                    time.sleep(0.3)
                    # 自适应俯冲：目标 z=0.58m（罐子顶端），不够低就多冲一点
                    ext_dist = max(0.02, p_curr[2] - 0.58) if p_curr is not None else 0.06
                    print(f"  自适应俯冲距离: {ext_dist*100:.1f} cm (当前z={p_curr[2]:.3f})")
                    node.smart_extrapolate_grasp(None, p_curr, ext_dist=ext_dist, lift_dist=0.15)
                else:
                    # 悬停超时或模型未触发夹爪：张爪后俯冲兜底
                    print("⚠️ 未确认夹住，张爪俯冲兜底...")
                    node.send_gripper_goal(0.0)
                    time.sleep(0.5)
                    node.smart_extrapolate_grasp(None, p_curr, ext_dist=0.12, lift_dist=0.15)

                print("✅ ACT 任务流圆满结束")
                node.go_home_direct()
            else:
                print(f"❌ 识别服务响应异常")
        except Exception as e:
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
