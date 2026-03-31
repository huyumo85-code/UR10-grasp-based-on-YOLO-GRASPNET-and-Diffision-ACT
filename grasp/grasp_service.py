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
    print("正在加载 GraspNet 模型到显存...")
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(os.path.join(GRASPNET_ROOT, "logs/log_kn/checkpoint.tar"), map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    model_net = net
    model_device = device
    
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        rgb_path = data.get('rgb_path')
        depth_path = data.get('depth_path')
        fx, fy = float(data.get('fx', 554.25)), float(data.get('fy', 554.25))
        cx, cy = float(data.get('cx', 320.0)), float(data.get('cy', 240.0))
        
        vis_save_path = data.get('vis_save_path', '/tmp/grasp_vis_result.png')

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        h, w = depth.shape

        # ================== 🌟 第一阶段：YOLO 实例分割推理 ==================
        yolo_results = yolo_net(rgb, verbose=False)
        detected_objects = []
        
        for r in yolo_results:
            # 严格检查：如果没有提取到多边形掩码(Mask)，直接跳过
            if r.masks is None:
                continue
                
            for i, box in enumerate(r.boxes):
                conf = float(box.conf[0].cpu().numpy())
                # 提高置信度门槛到 0.5，因为你的自训练模型非常准
                if conf < 0.5: continue  
                
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = yolo_net.names[cls_id]
                
                # 提取对应物体的多边形像素坐标矩阵
                polygon = r.masks.xy[i] 
                if len(polygon) == 0: continue
                
                detected_objects.append({
                    "obj_id": len(detected_objects) + 1,
                    "class_name": cls_name,
                    "polygon": polygon.tolist(), # 存入极其精确的边界点数组
                    "grasps": []
                })

        if len(detected_objects) == 0:
            cv2.imwrite(vis_save_path, yolo_results[0].plot())
            return jsonify({"status": "fail", "msg": "【YOLO 锅】视野中未检测到目标物体的实例分割掩码！"})

        # ================== 🌟 第二阶段：GraspNet 推理 (已接入 YOLO 掩码优化) ==================
        camera = CameraInfo(w, h, fx, fy, cx, cy, 1000.0)
        
        # 1. 创建一个与深度图同等尺寸的全黑掩码
        yolo_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 2. 将所有 YOLO 提取到的极其精确的物体多边形轮廓，在掩码图上填充为纯白 (255)
        for obj in detected_objects:
            pts = np.array(obj["polygon"], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(yolo_mask, [pts], 255)
            
        # 3. 核心魔法：使用掩码过滤深度图。背景、桌面等所有非目标物体的点云数据将被彻底抹除！
        depth_filtered = cv2.bitwise_and(depth, depth, mask=yolo_mask)

        cloud_organized = create_point_cloud_from_depth_image(depth_filtered, camera, True)
        mask = depth_filtered > 0
        cloud_masked = cloud_organized[mask]

        if len(cloud_masked) == 0:
            return jsonify({"status": "fail", "msg": "有效区域内无点云"})

        target_num = 20000
        idxs = np.random.choice(len(cloud_masked), target_num, replace=(len(cloud_masked) < target_num))
        cloud_sampled = cloud_masked[idxs]
        end_points = {'point_clouds': torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(model_device)}

        with torch.no_grad():
            end_points = model_net(end_points)
            grasp_preds = pred_decode(end_points)

        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        gg.nms()
        gg.sort_by_score()

        # ================== 🌟 第三阶段：高级多边形匹配算法 ==================
        for g in gg:
            if g.score < 0.05: continue
            x, y, z = g.translation
            if z < 0.1 or z > 2.5: continue
        
            u = int((x * fx) / z + cx)
            v = int((y * fy) / z + cy)
            if u < 0 or u >= w or v < 0 or v >= h: continue

            approach_vec = g.rotation_matrix[:, 0]
            cos_angle = np.dot(approach_vec, np.array([0, 0, 1])) 
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            closing_vec = g.rotation_matrix[:, 1]
            tilt_deg = np.degrees(np.arcsin(np.clip(abs(closing_vec[2]), 0.0, 1.0)))

            if angle_deg <= 45.0 and tilt_deg <= 30.0:
                grasp_dict = {
                    "translation": g.translation.tolist(),
                    "rotation": g.rotation_matrix.tolist(),
                    "score": float(g.score),
                    "width": float(g.width),
                    "angle": float(angle_deg),
                    "u": u, "v": v
                }
                
                # 核心魔法：使用 OpenCV 的 pointPolygonTest 判断抓取点是否真正落在多边形内部
                for obj in detected_objects:
                    pts_float = np.array(obj["polygon"], np.float32)
                    # 返回值 >= 0 表示该点在多边形的内部或边缘上
                    if cv2.pointPolygonTest(pts_float, (float(u), float(v)), False) >= 0:
                        if len(obj["grasps"]) < 5:
                            obj["grasps"].append(grasp_dict)
                        break 

        valid_objects = [obj for obj in detected_objects if len(obj["grasps"]) > 0]
        
        vis_img = rgb.copy()
        vis_img = draw_results(vis_img, detected_objects, fx, fy, cx, cy)
        cv2.imwrite(vis_save_path, vis_img)

        if valid_objects:
            return jsonify({"status": "success", "objects": valid_objects})
        else:
            return jsonify({"status": "fail", "msg": "【GraspNet 锅】YOLO 成功提取了掩码，但可乐罐表面没有符合 '自上而下' 角度条件的优质抓取点！"})

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n[🔥 Flask 服务内部崩溃]\n{err_msg}")
        return jsonify({"status": "fail", "msg": f"Flask内部报错: {str(e)}"}), 500

if __name__ == "__main__":
    load_models()
    app.run(host='0.0.0.0', port=PORT)
