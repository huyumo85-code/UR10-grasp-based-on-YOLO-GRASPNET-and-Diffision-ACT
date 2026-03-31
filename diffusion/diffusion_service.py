import sys
import os
import torch
import cv2
import numpy as np
import json
from collections import deque
from flask import Flask, request, jsonify
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torchvision import transforms
# 1. 动态获取路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from train_diffusion import DiffusionModel 

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载统计信息
STATS_PATH = os.path.join(os.path.expanduser('~/workspaces/ur_gz/robot_diffusion_data/'), 'stats.json')
with open(STATS_PATH, 'r') as f:
    stats = json.load(f)
    a_min = np.array(stats['action_min'])
    a_max = np.array(stats['action_max'])

# 3. 加载模型
# 🌟 指向新一轮训练好的 200 轮权重
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'diffusion_model_epoch_200.pth')
model = DiffusionModel().to(device)
if os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    print(f"✅ 成功加载权重: {MODEL_WEIGHTS_PATH}")
else:
    print(f"⚠️ 警告: 未找到权重文件 {MODEL_WEIGHTS_PATH}")
model.eval()

noise_scheduler = DDIMScheduler(num_train_timesteps=100)

# 全局历史记录滑动窗口
history_h = deque(maxlen=2)
history_g = deque(maxlen=2)
history_s = deque(maxlen=2)
val_transforms = transforms.Compose([
    transforms.ToPILImage(), # 先转回 PIL 格式
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def preprocess(img_bgr):
    # OpenCV 读进来是 BGR，我们要在把它变成 Tensor 之前转成 RGB 吗？
    # 🌟 回答是：不用转 RGB！因为你训练时录进 zarr 的就是 BGR！直接喂进去！
    
    # 利用统一的流水线处理
    img_t = val_transforms(img_bgr) 
    
    return img_t.unsqueeze(0).to(device)
@app.route('/reset', methods=['POST'])
def reset():
    history_h.clear()
    history_g.clear()
    history_s.clear()
    print("🧠 历史记忆已清空，准备开始新的轨迹...")
    return jsonify({"status": "success"})

@app.route('/get_action', methods=['POST'])
def get_action():
    try:
        file_h = request.files['image_hand'].read()
        file_g = request.files['image_global'].read()
        state_str = request.form.get('current_state')
        
        img_h_bgr = cv2.imdecode(np.frombuffer(file_h, np.uint8), cv2.IMREAD_COLOR)
        img_g_bgr = cv2.imdecode(np.frombuffer(file_g, np.uint8), cv2.IMREAD_COLOR)
        
        curr_state_raw = np.array(json.loads(state_str), dtype=np.float32)
        curr_state_norm = (curr_state_raw - a_min) / (a_max - a_min + 1e-6) * 2 - 1
        current_state_t = torch.from_numpy(curr_state_norm).unsqueeze(0).to(device).float()
        
        img_h_t = preprocess(img_h_bgr)
        img_g_t = preprocess(img_g_bgr)
        
        history_h.append(img_h_t)
        history_g.append(img_g_t)
        history_s.append(current_state_t)
        
        if len(history_h) == 1:
            history_h.append(img_h_t)
            history_g.append(img_g_t)
            history_s.append(current_state_t)

        seq_h = torch.stack(list(history_h), dim=1)
        seq_g = torch.stack(list(history_g), dim=1)
        seq_s = torch.stack(list(history_s), dim=1)
        
        with torch.no_grad():
            noisy_action = torch.randn((1, 16, 7)).to(device)
            # 🌟 致命修复：步数必须为 100，否则去噪方差崩溃
            noise_scheduler.set_timesteps(15) 
            
            for t in noise_scheduler.timesteps:
                t_tensor = t.unsqueeze(0).to(device).float()
                pred = model(seq_h, seq_g, seq_s, noisy_action, t_tensor)
                noisy_action = noise_scheduler.step(pred, t, noisy_action).prev_sample
        
        norm_action = noisy_action[0].cpu().numpy() 
        raw_action = (norm_action + 1) / 2 * (a_max - a_min + 1e-6) + a_min
        
        return jsonify({"action": raw_action.tolist()}) 
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, threaded=False)
