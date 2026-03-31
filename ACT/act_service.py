import os
import torch
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify
from lerobot.policies.act.modeling_act import ACTPolicy

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = os.path.expanduser("~/workspaces/ur_gz/ACT/outputs/train/2026-03-29/14-22-31_act/checkpoints/080000/pretrained_model")

print(f"正在加载 UR10 ACT 策略...")
policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH).to(device)
policy.eval()

@app.route('/reset', methods=['POST'])
def reset_policy():
    if hasattr(policy, 'reset'):
        policy.reset() # 必须清空，防止时序动作重叠爆炸
    print("🧠 内部时序队列已清空，准备执行新轨迹！")
    return jsonify({"status": "success"})

def preprocess(img_bgr):
    # 1. 颜色转换：cv2(BGR) -> 模型(RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 2. 缩放
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    # 3. 归一化与张量化
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return img_t.to(device)

@app.route('/get_action', methods=['POST'])
def get_action():
    try:
        file_h = request.files['image_hand'].read()   
        file_g = request.files['image_global'].read() 
        state_str = request.form.get('current_state') 
        
        img_h = cv2.imdecode(np.frombuffer(file_h, np.uint8), cv2.IMREAD_COLOR)
        img_g = cv2.imdecode(np.frombuffer(file_g, np.uint8), cv2.IMREAD_COLOR)
        
        # 依然保留图片调试，防相机接反
        cv2.imwrite("debug_high.png", img_g)
        cv2.imwrite("debug_low.png", img_h)

        # 🌟 直接使用 ROS 传过来的完美状态！不需要任何重新排序了！
        curr_state_ros = np.array(json.loads(state_str), dtype=np.float32)

        obs_dict = {
            "observation.images.cam_high": preprocess(img_g).unsqueeze(0), 
            "observation.images.cam_low": preprocess(img_h).unsqueeze(0),  
            "observation.state": torch.from_numpy(curr_state_ros).unsqueeze(0).to(device)
        }

        with torch.no_grad():
            action_output = policy.select_action(obs_dict)
            action_model = action_output[0].cpu().numpy() 

        # 🌟 直接返回模型预测的动作！完美 1:1 对应，不需要翻译了！
        action_ros = [float(x) for x in action_model]

        return jsonify({"action": action_ros}) 
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, threaded=True)
