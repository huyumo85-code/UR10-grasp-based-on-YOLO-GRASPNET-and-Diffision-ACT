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

# 你录制时使用的绝对起点
HOME_JOINTS = [-0.28, -1.61, 0.91, -0.84, -1.57, -0.24]

class TemporalEnsembler:
    def __init__(self, horizon=16):
        self.horizon = horizon
        self.preds = []
        self.step = 0

    def update_and_get(self, new_traj, lookahead=1):
        self.preds.append((self.step, np.array(new_traj)))
        self.preds = [p for p in self.preds if p[0] + self.horizon > self.step + lookahead]
        
        current_actions = []
        weights = []
        for p_step, traj in self.preds:
            idx = (self.step + lookahead) - p_step
            if 0 <= idx < self.horizon:
                current_actions.append(traj[idx])
                weights.append(np.exp(-idx * 0.2)) 
        if not current_actions:
            return np.array(new_traj[0]) # 兜底逻辑
        weights = np.array(weights) / np.sum(weights)
        ensembled_action = np.average(current_actions, axis=0, weights=weights)
        
        self.step += 1
        return ensembled_action

class RosDiffusionManager(Node):
    def __init__(self):
        super().__init__(
            'ros_diffusion_manager',
            allow_undeclared_parameters=True, 
            automatically_declare_parameters_from_overrides=True
        )
        self.bridge = CvBridge()
        self.latest_rgb = None        
        self.latest_global_rgb = None 
        self.vis_img = None 
        self.current_joint_pos = None 

        # 🌟 核心新增：夹爪状态锁存器与防连发记录器
        self.is_gripping = False 
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

    def send_gripper_goal(self, position):
        # 🌟 防连发机制：只有当指令发生改变时，才向服务器发送，防止冲爆底层驱动
        if self.last_sent_grip_cmd == position:
            return 
            
        self.last_sent_grip_cmd = position
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 100.0 
        
        if not self._gripper_action_client.wait_for_server(timeout_sec=0.1): 
            return
            
        self._gripper_action_client.send_goal_async(goal_msg)

    def img_cb(self, m): self.latest_rgb = m
    def global_img_cb(self, m): self.latest_global_rgb = m

    def joint_cb(self, msg):
        pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
        try:
            temp_pos = [pos_dict[name] for name in self.arm_joint_names]
            temp_pos.append(pos_dict.get(self.gripper_joint_name, 0.0))
            self.current_joint_pos = temp_pos
        except KeyError: pass
        
    def go_home_direct(self):
        print("🏠 正在强制复位到起跑线 (Home)...")
        # 🌟 每次回起点时，重置锁存状态
        self.is_gripping = False
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

    # 🌟 新增：批量发送连续轨迹的函数
    def publish_action(self, action): 
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = self.get_clock().now().to_msg()
        arm_msg.joint_names = self.arm_joint_names
        
        arm_p = JointTrajectoryPoint()
        arm_p.positions = [float(x) for x in action[:6]]
        
        # 🌟 核心：计算速度期望，严防 MoveIt 急刹车导致抽搐
        if self.current_joint_pos is not None:
            # 我们让它去追 0.2 秒后的目标，所以除以 0.2 算出大概速度
            vels = [(float(x) - cur) / 0.2 for x, cur in zip(action[:6], self.current_joint_pos[:6])]
            arm_p.velocities = [float(np.clip(v, -1.0, 1.0)) for v in vels]
        else:
            arm_p.velocities = [0.0] * 6
            
        # 抛出 0.2 秒外的“胡萝卜”
        arm_p.time_from_start.sec = 0
        arm_p.time_from_start.nanosec = 200_000_000 
        
        arm_msg.points.append(arm_p)
        self.joint_pub.publish(arm_msg)

        # 夹爪意图保持原样
        target_grip = float(action[6])
        if target_grip > 0.65:
            self.is_gripping = True
            
        if self.is_gripping:
            self.send_gripper_goal(0.8)
        else:
            self.send_gripper_goal(0.0)
    def wait_for_completion(self, target_pos, timeout=1.0):
        start_time = self.get_clock().now()
        rate = self.create_rate(50) 
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
            if self.current_joint_pos is not None:
                diff = np.abs(np.array(self.current_joint_pos[:6]) - np.array(target_pos[:6]))
                if np.max(diff) < 0.02: return True 
            rate.sleep()
        return False

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
        input("\n>>> [按回车] 启动全过程端到端抓取 (Diffusion)...")
        if node.latest_rgb is None or node.latest_global_rgb is None:
            print("❌ 错误：相机数据未就绪")
            continue
        
        #node.go_home_direct()

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
                print(f"🚀 轨迹执行中 (启用 Action Chunking 算法)...")
                
                requests.post("http://127.0.0.1:5006/reset")
                # 🌟 重新启用平滑器！(因为模型已经被特训过，不会再发生手腕僵硬了)
                ensembler = TemporalEnsembler(horizon=16)
                node.is_gripping = False
                
                # 回归一拍一动的 300 步循环
                for i in range(300): 
                    loop_start = time.time() # 🌟 记录起点
                    
                    if not rclpy.ok(): break
                    img_h = node.bridge.imgmsg_to_cv2(node.latest_rgb, "bgr8")
                    img_g = node.bridge.imgmsg_to_cv2(node.latest_global_rgb, "bgr8")
                    _, enc_h = cv2.imencode('.png', img_h)
                    _, enc_g = cv2.imencode('.png', img_g)
                    current_state_data = json.dumps(node.current_joint_pos)
                    
                    act_resp = requests.post("http://127.0.0.1:5006/get_action", 
                                            files={'image_hand': enc_h.tobytes(), 'image_global': enc_g.tobytes()},
                                            data={'current_state': current_state_data})
                    
                    if act_resp.status_code == 200:
                        action_traj = act_resp.json().get('action') 
                        if action_traj:
                            # 🌟 取出未来 0.2 秒的平滑轨迹点
                            smooth_action = ensembler.update_and_get(action_traj, lookahead=2)
                            node.publish_action(smooth_action)
                            
                            # 每 5 步打印一次，方便观察
                            if i % 5 == 0: 
                                lock_status = "🔒 已锁死闭合" if node.is_gripping else "等待触发"
                                target_arm = smooth_action[:6]
                                curr_arm = node.current_joint_pos[:6] if node.current_joint_pos else [0.0] * 6
                                str_target = ", ".join([f"{x: .3f}" for x in target_arm])
                                str_curr = ", ".join([f"{x: .3f}" for x in curr_arm])
                                
                                print(f"📍 Progress: {i:03d}/300 | 夹爪意图: {smooth_action[6]:.3f} ({lock_status})")
                                print(f"   🎯 [模型目标 J1-J6]: {str_target}")
                                print(f"   🦾 [物理当前 J1-J6]: {str_curr}")
                                print("-" * 60)
                        else: break
                    else: break
                    
                    # 🌟 灵魂核心：严格把控 0.1 秒的真实流速！
                    elapsed = time.time() - loop_start
                    sleep_time = max(0.0, 0.1 - elapsed) 
                    time.sleep(sleep_time)
                print("✅ 任务流运行完毕")
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
