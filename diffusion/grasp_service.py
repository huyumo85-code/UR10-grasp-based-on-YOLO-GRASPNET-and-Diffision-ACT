import sys
import os
import numpy as np
import cv2
import json
import torch
import traceback
from flask import Flask, request, jsonify
from graspnetAPI import GraspGroup
from ultralytics import YOLO

# ================= 配置区域 =================
GRASPNET_ROOT = "/home/hym/graspnet-baseline"
PORT = 5005
# 🌟 已替换为你新鲜出炉的 Gazebo 专属实例分割模型
MODEL_PATH = '/home/hym/下载/can_seg_my/runs/can_segmentation/weights/best.pt'
# ===========================================

sys.path.append(GRASPNET_ROOT)
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'dataset'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

from graspnet import GraspNet, pred_decode
from data_utils import CameraInfo, create_point_cloud_from_depth_image

app = Flask(__name__)

model_net = None
model_device = None
yolo_net = None

def load_models():
    global model_net, model_device, yolo_net
    
    
    print("正在加载 YOLO 实例分割模型...")
    yolo_net = YOLO(MODEL_PATH)
    print("✅ 双模型加载完成，服务准备就绪。")

def project_point(p3d, fx, fy, cx, cy):
    x, y, z = p3d
    if z <= 0.01: return None
    u = int(x * fx / z + cx)
    v = int(y * fy / z + cy)
    return (u, v)

# 🌟 重构绘图逻辑：绘制不规则多边形轮廓
def draw_results(img, objects_data, fx, fy, cx, cy):
    for obj in objects_data:
        obj_id = obj['obj_id']
        cls_name = obj['class_name']
        polygon = obj.get('polygon', [])
        
        # 1. 绘制带有透明度的多边形 Mask 边缘
        if len(polygon) > 0:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            # 沿着可乐罐画出极其贴合的异形轮廓线
            cv2.polylines(img, [pts], isClosed=True, color=(255, 144, 30), thickness=3)
            
            # 寻找多边形最高点来写字，防止标签重叠
            top_y_idx = np.argmin(pts[:, 0, 1])
            text_pos = (pts[top_y_idx, 0, 0], max(pts[top_y_idx, 0, 1] - 10, 10))
            cv2.putText(img, f"Obj {obj_id}: {cls_name}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 144, 30), 2)

        # 2. 绘制落在该物体精确轮廓内的抓取点
        for i, item in enumerate(obj['grasps']):
            try:
                center_3d = np.array(item["translation"])
                rot = np.array(item["rotation"]) 
                width = item.get("width", 0.08)  
                
                jaw1_3d = center_3d + rot[:, 1] * (width / 2)
                jaw2_3d = center_3d - rot[:, 1] * (width / 2)
                wrist_3d = center_3d - rot[:, 0] * 0.08 

                pt_center = project_point(center_3d, fx, fy, cx, cy)
                pt_jaw1 = project_point(jaw1_3d, fx, fy, cx, cy)
                pt_jaw2 = project_point(jaw2_3d, fx, fy, cx, cy)
                pt_wrist = project_point(wrist_3d, fx, fy, cx, cy)

                if pt_center:
                    cv2.circle(img, pt_center, 6, (0, 0, 255), -1) 
                    cv2.putText(img, f"#{obj_id}-{i+1}", (pt_center[0]+15, pt_center[1]-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                if pt_jaw1 and pt_jaw2:
                    cv2.line(img, pt_jaw1, pt_jaw2, (0, 255, 255), 4)
                    cv2.circle(img, pt_jaw1, 5, (0, 255, 0), -1)
                    cv2.circle(img, pt_jaw2, 5, (0, 255, 0), -1)

                if pt_center and pt_wrist:
                    cv2.arrowedLine(img, pt_wrist, pt_center, (0, 0, 255), 4, tipLength=0.2)
            except Exception as e:
                print(f"绘图错误: {e}")
                continue
    return img

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        # 🌟 修改 1：接收手眼相机的图片
        if 'image_hand' in request.files:
            file = request.files['image_hand'].read()
        elif 'image' in request.files:
            file = request.files['image'].read()
        else:
            return jsonify({"status": "error", "msg": "未找到图像输入，请确保 ROS 端发送了 image_hand"})

        rgb = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        
        # 🌟 修改 2：增加可视化调试（运行后查看该路径下的图片）
        # 看看是不是相机离得太近了只看到一片白，或者根本没对准罐子
        debug_path = os.path.join(os.path.expanduser("~"), "hand_eye_debug.png")
        cv2.imwrite(debug_path, rgb)
        
        # 🌟 修改 3：YOLO 推理 (降低 conf 阈值)
        # 如果依然是 0，可以尝试改为 conf=0.15
        results = yolo_net(rgb, conf=0.25) 
        
        detected_objects = []
        for i, r in enumerate(results):
            if r.masks is None: continue
            for j, mask in enumerate(r.masks.xy):
                poly = mask.astype(int).tolist() 
                detected_objects.append({
                    "obj_id": len(detected_objects),
                    "class_name": yolo_net.names[int(r.boxes.cls[j])],
                    "polygon": poly,
                    "confidence": float(r.boxes.conf[j])
                })
        
        # 打印日志方便查看服务端状态
        print(f"📷 收到识别请求 | 识别结果: {len(detected_objects)} 个目标")
        
        return jsonify({"status": "success", "objects": detected_objects})
    except Exception as e:
        traceback.print_exc() # 在控制台打印详细报错
        return jsonify({"status": "error", "msg": str(e)})

if __name__ == "__main__":
    load_models()
    app.run(host='0.0.0.0', port=PORT)
