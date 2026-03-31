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

    # 🌟 修改 1：将前瞻调小为 2，让机械臂紧紧贴合当前预测轨迹，减少超调
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
    def send_gripper_goal(self, position):
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
            
            # 🌟 新增：打印当前关节信息（弧度），每 10 帧打印一次避免刷屏
            if getattr(self, '_log_counter', 0) % 50 == 0:
                joint_str = ", ".join([f"{p:.2f}" for p in temp_pos[:6]])
                self.get_logger().info(f"📊 当前关节角 (Rad): [{joint_str}] | 夹爪: {temp_pos[6]:.2f}")
            self._log_counter = getattr(self, '_log_counter', 0) + 1
            
        except KeyError: pass
        
    def go_home_direct(self):
        print("🏠 正在强制复位到起跑线 (Home)...")
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
            diff = target_arm - current_arm
            max_diff = np.max(np.abs(diff))
            
            # 🌟 修复核心 1：将 0.15 放宽到 1.0！
            # 0.5秒内移动 1.0 弧度（57度）是安全的，不要让机械臂总是掉队！
            max_allowed = 0.3
            
            if max_diff > max_allowed:
                limited_diff = diff * (max_allowed / max_diff)
            else:
                limited_diff = diff
                
            arm_p.positions = [float(x) for x in (current_arm + limited_diff)]
        else:
            arm_p.positions = [float(x) for x in action[:6]]
            
        # 🌟 修改 3：清空速度要求，并给控制器 150ms 缓冲，防止急刹车发抖
        arm_p.velocities = [] 
        arm_p.accelerations = []
        arm_p.time_from_start.nanosec = 100_000_000 
        arm_msg.points.append(arm_p)
        self.joint_pub.publish(arm_msg)

        target_grip = float(action[6])
        if target_grip > 0.4:
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
                print(f"🚀 ACT 轨迹执行中 (直接下发模式)...")
                
                # 🌟 发送重置信号，清空 LeRobot 的内部记忆队列
                requests.post("http://127.0.0.1:5006/reset")
                
                for i in range(300): # 增加一点最大步数容错
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
                        action_step = act_resp.json().get('action') 
                        if action_step:
                            # 🌟 修复核心：LeRobot 返回的就是单步动作，直接发给机械臂！
                            node.publish_action(action_step)
                            time.sleep(0.1)
                            
                            # 打印进度，心里有底
                            if i % 2 == 0: 
                                print(f"  Progress: {i}/300 | Grip_Pred: {action_step[6]:.2f}")
                        else: break
                    else: break
                print("✅ act任务流运行完毕")
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
