import subprocess
import time
import os
import random  # 引入随机，让罐子落点更自然

# ================= 配置区域 =================
# 1. 模型路径配置
MODEL_ROOT_DIR = os.path.expanduser("~/下载/gazebo_models-master")
COKE_SDF_PATH = os.path.join(MODEL_ROOT_DIR, "coke_can", "model.sdf")

# 2. 桌子配置
TABLE_W, TABLE_L, TABLE_H = 0.6, 1.0, 0.4
TABLE_X = 0.6   
TABLE_Y = 0.0
POSE_TABLE_Z = TABLE_H / 2 

# 3. 盒子配置
BIN_W, BIN_L, BIN_H = 0.4, 0.6, 0.15
WALL_THICK = 0.01
SURFACE_Z = TABLE_H 
POSE_BIN_Z = SURFACE_Z + (BIN_H / 2) + 0.01

# 4. 生成高度 (基础高度)
DROP_Z_BASE = SURFACE_Z + BIN_H + 0.1
# ===========================================

def create_sdf_file(filename, content):
    path = f"/tmp/{filename}.sdf"
    with open(path, "w") as f:
        f.write(content)
    return path

def spawn_entity(name, file_path, x, y, z, roll=0, pitch=0, yaw=0):
    print(f"📦 正在生成: {name} ...")
    cmd = [
        "ros2", "run", "ros_gz_sim", "create",
        "-file", file_path,
        "-name", name,
        "-x", str(x), "-y", str(y), "-z", str(z),
        "-R", str(roll), "-P", str(pitch), "-Y", str(yaw)
    ]
    subprocess.run(cmd)
    time.sleep(0.2) # 稍微缩短间隔，加快生成速度

# --- SDF 定义 ---
table_sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6"><model name="table_model"><static>true</static><link name="link"><collision name="collision"><geometry><box><size>{TABLE_W} {TABLE_L} {TABLE_H}</size></box></geometry></collision><visual name="visual"><geometry><box><size>{TABLE_W} {TABLE_L} {TABLE_H}</size></box></geometry><material><ambient>0.5 0.3 0.1 1</ambient><diffuse>0.5 0.3 0.1 1</diffuse></material></visual></link></model></sdf>"""

bin_sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6"><model name="bin_model"><static>true</static><link name="link">
<collision name="bottom"><pose>0 0 {-BIN_H/2 + WALL_THICK/2} 0 0 0</pose><geometry><box><size>{BIN_W} {BIN_L} {WALL_THICK}</size></box></geometry></collision><visual name="bottom"><pose>0 0 {-BIN_H/2 + WALL_THICK/2} 0 0 0</pose><geometry><box><size>{BIN_W} {BIN_L} {WALL_THICK}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<collision name="front"><pose>{BIN_W/2 - WALL_THICK/2} 0 0 0 0 0</pose><geometry><box><size>{WALL_THICK} {BIN_L} {BIN_H}</size></box></geometry></collision><visual name="front"><pose>{BIN_W/2 - WALL_THICK/2} 0 0 0 0 0</pose><geometry><box><size>{WALL_THICK} {BIN_L} {BIN_H}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<collision name="back"><pose>{-BIN_W/2 + WALL_THICK/2} 0 0 0 0 0</pose><geometry><box><size>{WALL_THICK} {BIN_L} {BIN_H}</size></box></geometry></collision><visual name="back"><pose>{-BIN_W/2 + WALL_THICK/2} 0 0 0 0 0</pose><geometry><box><size>{WALL_THICK} {BIN_L} {BIN_H}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<collision name="left"><pose>0 {BIN_L/2 - WALL_THICK/2} 0 0 0 0</pose><geometry><box><size>{BIN_W} {WALL_THICK} {BIN_H}</size></box></geometry></collision><visual name="left"><pose>0 {BIN_L/2 - WALL_THICK/2} 0 0 0 0</pose><geometry><box><size>{BIN_W} {WALL_THICK} {BIN_H}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<collision name="right"><pose>0 {-BIN_L/2 + WALL_THICK/2} 0 0 0 0</pose><geometry><box><size>{BIN_W} {WALL_THICK} {BIN_H}</size></box></geometry></collision><visual name="right"><pose>0 {-BIN_L/2 + WALL_THICK/2} 0 0 0 0</pose><geometry><box><size>{BIN_W} {WALL_THICK} {BIN_H}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
</link></model></sdf>"""

if __name__ == "__main__":
    current_resource_path = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = f"{MODEL_ROOT_DIR}:{current_resource_path}"

    if not os.path.exists(COKE_SDF_PATH):
        print(f"❌ 错误：找不到文件 {COKE_SDF_PATH}")
        exit(1)

    print("🚀 正在构建 可乐罐的纯净场景...")
    
    # 1. 生成环境基础
    p_table = create_sdf_file("my_table", table_sdf)
    p_bin = create_sdf_file("my_bin", bin_sdf)
    
    spawn_entity("table", p_table, TABLE_X, TABLE_Y, POSE_TABLE_Z)
    spawn_entity("bin", p_bin, TABLE_X, TABLE_Y, POSE_BIN_Z)

    # 2. 循环生成 10 个可乐罐
    for i in range(1, 8):
        # 在盒子范围内随机微调坐标 (盒子 W=0.4, L=0.6)
        off_x = random.uniform(-0.1, 0.1)
        off_y = random.uniform(-0.15, 0.15)
        # 增加 Z 高度，让它们依次落下
        drop_z = DROP_Z_BASE + (i * 0.15)
        
        # 随机姿态：有些立着，有些横着
        r = random.choice([0, 1.57]) # 0 是立着，1.57 是躺着
        
        spawn_entity(f"coke_{i}", COKE_SDF_PATH, TABLE_X + off_x, TABLE_Y + off_y, drop_z, roll=r)

    print(f"✅ 10 个可乐罐已全部投放！")
