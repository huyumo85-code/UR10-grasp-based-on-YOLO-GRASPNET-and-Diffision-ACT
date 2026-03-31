import subprocess
import time
import os

# ================= 配置区域 =================
# 1. 模型路径配置
MODEL_ROOT_DIR = os.path.expanduser("~/下载/gazebo_models-master")
COKE_SDF_PATH = os.path.join(MODEL_ROOT_DIR, "coke_can", "model.sdf")

# 2. 桌子配置 (修改这里！)
TABLE_W, TABLE_L, TABLE_H = 0.6, 1.0, 0.4

# --- 关键修改 ---
# 之前是 0.8，改为 0.6，让桌子靠近机器人
# 注意：不要小于 0.35，否则桌子会和机器人基座撞在一起
TABLE_X = 0.6   
TABLE_Y = 0.0
# ----------------

POSE_TABLE_Z = TABLE_H / 2 

# 3. 盒子配置
BIN_W, BIN_L, BIN_H = 0.4, 0.6, 0.15
WALL_THICK = 0.01
SURFACE_Z = TABLE_H 
POSE_BIN_Z = SURFACE_Z + (BIN_H / 2) + 0.01

# 4. 可乐罐生成高度
DROP_Z = SURFACE_Z + BIN_H + 0.15
# ===========================================

def create_sdf_file(filename, content):
    """生成临时的 SDF 文件"""
    path = f"/tmp/{filename}.sdf"
    with open(path, "w") as f:
        f.write(content)
    return path

def spawn_entity(name, file_path, x, y, z, roll=0, pitch=0, yaw=0):
    """调用 ROS2 命令生成物体"""
    print(f"📦 正在生成: {name} ...")
    cmd = [
        "ros2", "run", "ros_gz_sim", "create",
        "-file", file_path,
        "-name", name,
        "-x", str(x), "-y", str(y), "-z", str(z),
        "-R", str(roll), "-P", str(pitch), "-Y", str(yaw)
    ]
    subprocess.run(cmd)
    time.sleep(0.5) 

# --- SDF 定义 ---
# 桌子已经是 static=true
table_sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6"><model name="table_model"><static>true</static><link name="link"><collision name="collision"><geometry><box><size>{TABLE_W} {TABLE_L} {TABLE_H}</size></box></geometry></collision><visual name="visual"><geometry><box><size>{TABLE_W} {TABLE_L} {TABLE_H}</size></box></geometry><material><ambient>0.5 0.3 0.1 1</ambient><diffuse>0.5 0.3 0.1 1</diffuse></material></visual></link></model></sdf>"""

# ================= 核心修改处 =================
# 将盒子的 <static>false</static> 改成了 <static>true</static>
bin_sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6"><model name="bin_model"><static>true</static><link name="link"><inertial><mass>0.5</mass><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
<collision name="bottom"><pose>0 0 {-BIN_H/2 + WALL_THICK/2} 0 0 0</pose><geometry><box><size>{BIN_W} {BIN_L} {WALL_THICK}</size></box></geometry></collision><visual name="bottom"><pose>0 0 {-BIN_H/2 + WALL_THICK/2} 0 0 0</pose><geometry><box><size>{BIN_W} {BIN_L} {WALL_THICK}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<collision name="front"><pose>{BIN_W/2 - WALL_THICK/2} 0 0 0 0 0</pose><geometry><box><size>{WALL_THICK} {BIN_L} {BIN_H}</size></box></geometry></collision><visual name="front"><pose>{BIN_W/2 - WALL_THICK/2} 0 0 0 0 0</pose><geometry><box><size>{WALL_THICK} {BIN_L} {BIN_H}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<collision name="back"><pose>{-BIN_W/2 + WALL_THICK/2} 0 0 0 0 0</pose><geometry><box><size>{WALL_THICK} {BIN_L} {BIN_H}</size></box></geometry></collision><visual name="back"><pose>{-BIN_W/2 + WALL_THICK/2} 0 0 0 0 0</pose><geometry><box><size>{WALL_THICK} {BIN_L} {BIN_H}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<collision name="left"><pose>0 {BIN_L/2 - WALL_THICK/2} 0 0 0 0</pose><geometry><box><size>{BIN_W} {WALL_THICK} {BIN_H}</size></box></geometry></collision><visual name="left"><pose>0 {BIN_L/2 - WALL_THICK/2} 0 0 0 0</pose><geometry><box><size>{BIN_W} {WALL_THICK} {BIN_H}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<collision name="right"><pose>0 {-BIN_L/2 + WALL_THICK/2} 0 0 0 0</pose><geometry><box><size>{BIN_W} {WALL_THICK} {BIN_H}</size></box></geometry></collision><visual name="right"><pose>0 {-BIN_L/2 + WALL_THICK/2} 0 0 0 0</pose><geometry><box><size>{BIN_W} {WALL_THICK} {BIN_H}</size></box></geometry><material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
</link></model></sdf>"""

# ================= 新增：简易香蕉 SDF =================
# 使用一个细长的圆柱体代替，长度0.15m，半径0.02m，颜色为纯黄色
banana_sdf = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="banana_model">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>0.15</mass>
        <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.02</radius><length>0.15</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.02</radius><length>0.15</length></cylinder>
        </geometry>
        <material>
          <ambient>1.0 1.0 0.0 1</ambient>
          <diffuse>1.0 1.0 0.0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
# =======================================================
# ================= 新增：10cm 红色立方体 SDF =================
cube_sdf = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="cube_model">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>0.2</mass>
        <inertia ixx="0.00033" ixy="0" ixz="0" iyy="0.00033" iyz="0" izz="0.00033"/>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.07 0.07 0.07</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.07 0.07 0.07</size></box>
        </geometry>
        <material>
          <ambient>0.8 0.1 0.1 1</ambient>
          <diffuse>0.8 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
# =======================================================

if __name__ == "__main__":
    # 设置环境变量
    current_resource_path = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = f"{MODEL_ROOT_DIR}:{current_resource_path}"
    print(f"🌍 模型搜索路径已设置: {MODEL_ROOT_DIR}")

    if not os.path.exists(COKE_SDF_PATH):
        print(f"❌ 错误：找不到文件 {COKE_SDF_PATH}")
        exit(1)

    print("🚀 开始构建测试场景...")
    
    # 1. 生成桌子、盒子和香蕉的临时SDF文件
    p_table = create_sdf_file("my_table", table_sdf)
    p_bin = create_sdf_file("my_bin", bin_sdf)
    p_banana = create_sdf_file("my_banana", banana_sdf)  # <--- 新增
    p_cube = create_sdf_file("my_cube", cube_sdf)  # <--- 新增立方体
    # 2. 生成环境
    spawn_entity("table", p_table, TABLE_X, TABLE_Y, POSE_TABLE_Z)
    spawn_entity("bin", p_bin, TABLE_X, TABLE_Y, POSE_BIN_Z)
    
    # 3. 生成 3 个真实可乐罐
    spawn_entity("coke_1", COKE_SDF_PATH, TABLE_X, TABLE_Y, DROP_Z)
    spawn_entity("coke_2", COKE_SDF_PATH, TABLE_X + 0.08, TABLE_Y + 0.08, DROP_Z + 0.1)
    spawn_entity("coke_3", COKE_SDF_PATH, TABLE_X - 0.08, TABLE_Y - 0.08, DROP_Z + 0.2, roll=1.57)
    spawn_entity("coke_4", COKE_SDF_PATH, TABLE_X + 0.08, TABLE_Y + 0.08, DROP_Z + 0.25, roll=1.57)
    spawn_entity("coke_5", COKE_SDF_PATH, TABLE_X - 0.08, TABLE_Y - 0.08, DROP_Z + 0.3, roll=1.57)
    # 4. 生成 3 个简易香蕉 (黄色细长圆柱体)
    # 注意：生成高度 (DROP_Z + 0.3) 比可乐罐更高，防止它们同时生成在同一个位置发生物理爆炸
    print("🍌 正在生成简易香蕉...")
    spawn_entity("banana_1", p_banana, TABLE_X + 0.05, TABLE_Y - 0.05, DROP_Z + 0.3, pitch=1.57)
    spawn_entity("banana_2", p_banana, TABLE_X - 0.05, TABLE_Y + 0.05, DROP_Z + 0.4, pitch=1.57, yaw=1.0)
    spawn_entity("banana_3", p_banana, TABLE_X, TABLE_Y + 0.1, DROP_Z + 0.5, roll=0.5, pitch=1.57)
    # 5. 生成 2 个红色的 7cm 立方体
    print("🧊 正在生成 7cm 立方体...")
    # 故意把 DROP_Z 设置得更高，让它们最后落下
    spawn_entity("cube_1", p_cube, TABLE_X + 0.1, TABLE_Y + 0.05, DROP_Z + 0.6)
    spawn_entity("cube_2", p_cube, TABLE_X - 0.1, TABLE_Y - 0.05, DROP_Z + 0.7)
	
    print("✅ 场景生成完毕！(桌子已拉近，包含可乐罐与简易香蕉)")
