#!/bin/bash

# ================= 配置区域 =================
# 工作空间路径
WORKSPACE_DIR=~/workspaces/ur_gz
# Gazebo模型路径 (请确认此路径正确)
GAZEBO_MODELS=$HOME/下载/gazebo_models-master
WORKSPACE_SHARE=$WORKSPACE_DIR/install/share
# 公共的环境初始化命令
SETUP_CMD="source /opt/ros/humble/setup.bash && source $WORKSPACE_DIR/install/setup.bash"
# ===========================================

echo "🚀 正在启动 UR10 仿真系统 (包含自动控制)..."

# 1. 启动 Gazebo 仿真 (终端 1)
gnome-terminal --tab --title="1_Gazebo_Sim" -- bash -c "
echo '启动仿真环境...';
$SETUP_CMD;
# 在终端 1 的脚本里修改
export WORKSPACE_SHARE=$WORKSPACE_DIR/install/share
export IGN_GAZEBO_RESOURCE_PATH=$GAZEBO_MODELS:$WORKSPACE_SHARE:\$IGN_GAZEBO_RESOURCE_PATH
# 【修改 1】：在 ros2 launch 后面直接加 use_sim_time:=true
ros2 launch ur_simulation_gz ur_sim_control.launch.py \
 ur_type:=ur10 \
 description_package:=my_ur_gripper_config \
 description_file:=ur10_robotiq.urdf.xacro \
 controllers_file:=my_controllers.yaml \
 runtime_config_package:=ur_simulation_gz \
 launch_rviz:=false \
 use_sim_time:=true;
exec bash"

# 2. 激活控制器 & MoveIt (终端 2)
# 延时 8 秒
gnome-terminal --tab --title="2_MoveIt" -- bash -c "
$SETUP_CMD;
echo '等待仿真启动 (8s)...'; sleep 8;

echo '✅ 激活夹爪控制器 (主臂控制器已由Gazebo自动激活)...';
ros2 control load_controller robotiq_gripper_controller --set-state active;

echo '🚀 启动 MoveIt...';
# 这里你原本就加了，保持不变即可
ros2 launch ur10_gripper_moveit_config moveit.launch.py use_sim_time:=true;
exec bash"

# 3. 桥接节点 (终端 3)
gnome-terminal --tab --title="3_Bridge" -- bash -c "
$SETUP_CMD;
echo '启动 ROS-Gazebo 桥接 (QoS 修复)...';sleep 12;
# 【修改 2】：使用 ros2 run 时，需要加上 --ros-args -p
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=$HOME/workspaces/ur_gz/bridge_config.yaml -p use_sim_time:=true;
exec bash"

# 4. 运行图像处理脚本 (终端 4)
gnome-terminal --tab --title="4_Camera_Script" -- bash -c "
$SETUP_CMD;
echo '等待桥接就绪 (12s)...'; sleep 16;
echo '运行图像脚本...';
# 【修改 3】：给 Python 节点传递 ROS 参数
/usr/bin/python3 ~/workspaces/ur_gz/camera_image.py --ros-args -p use_sim_time:=true;
exec bash"

# ================= 新增部分 =================
# 5. 生成物体并执行控制 (终端 5)
gnome-terminal --tab --title="5_Spawn_Ctrl" -- bash -c "
$SETUP_CMD;
export IGN_GAZEBO_RESOURCE_PATH=$GAZEBO_MODELS:\$IGN_GAZEBO_RESOURCE_PATH;
echo '等待环境完全就绪 (15s)...'; sleep 20;
echo '切换到工作目录...';
cd ~/workspaces/ur_gz/creat_coke;
echo '1. 生成可乐罐...';
# 生成脚本如果不涉及监听TF或订阅Topic，可以不加；为了保险加上也可以
/usr/bin/python3 spawn_cokes.py --ros-args -p use_sim_time:=true;
echo '等待物体稳定 (5s)...'; 
sleep 15;
exec bash"

# 6. 移动到初始位置 (终端 6)
gnome-terminal --tab --title="6_11" -- bash -c "
$SETUP_CMD;
echo '等待仿真启动 (20s)...'; sleep 30;
echo '2. 执行移动控制脚本 (11.py)...';
cd ~/workspaces/ur_gz
# 【修改 4】：控制脚本也必须用仿真时间，否则可能因为 TF 过期报错
/usr/bin/python3 11.py --ros-args -p use_sim_time:=true;
echo '✨ 环境部署完毕！请在新终端中运行:';
exec bash"

echo "✨ 所有服务已启动！请检查各终端是否有报错。下一终端python3 ~/workspaces/ur_gz/diffusion/ros_diffusion_mgr.py --ros-args -p use_sim_time:=true"
