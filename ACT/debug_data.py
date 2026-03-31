import numpy as np

# 自动处理新老版本导入路径
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

print("正在加载已转换的 LeRobot 数据集...")
# 直接读取你之前转换好的本地数据集
dataset = LeRobotDataset("hym/ur10_act_dataset")

# 抓取第一帧的数据
frame = dataset[0]

print("\n--- 1. 检查图像形状 (防通道错位) ---")
print(f"cam_high 形状: {frame['observation.images.cam_high'].shape}")
print(f"cam_low 形状:  {frame['observation.images.cam_low'].shape}")

print("\n--- 2. 检查机械臂状态 Qpos [第一帧] ---")
print(np.round(frame['observation.state'].numpy(), 3))

print("\n--- 3. 检查期望动作 Action [第一帧] ---")
print(np.round(frame['action'].numpy(), 3))
