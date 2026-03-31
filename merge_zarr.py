import zarr
import os
import numpy as np
import json

# 设置路径
src_dir = os.path.expanduser("~/workspaces/ur_gz/robot_diffusion_data/zarr_data")
output_path = os.path.expanduser("~/workspaces/ur_gz/robot_diffusion_data/train_dataset.zarr")

all_img_hand = []
all_img_global = []
all_actions = []
episode_ends = []

curr_idx = 0
files = sorted([f for f in os.listdir(src_dir) if f.endswith('.zarr')])

print(f"正在处理 {len(files)} 个演示文件...")

for f in files:
    z = zarr.open(os.path.join(src_dir, f), mode='r')
    
    # 读取数据
    img_h = z['data/obs/img_hand_eye'][:]
    img_g = z['data/obs/img_global'][:]
    act = z['data/action'][:]
    
    all_img_hand.append(img_h)
    all_img_global.append(img_g)
    all_actions.append(act)
    
    # 记录每个 episode 的结束位置
    curr_idx += len(act)
    episode_ends.append(curr_idx)

# 合并为大数组
full_img_hand = np.concatenate(all_img_hand, axis=0)
full_img_global = np.concatenate(all_img_global, axis=0)
full_actions = np.concatenate(all_actions, axis=0)

# 写入统一的 Zarr
root = zarr.open(output_path, mode='w')
data_grp = root.create_group('data')
obs_grp = data_grp.create_group('obs')

obs_grp.create_dataset('img_hand_eye', data=full_img_hand, chunks=(1, *full_img_hand.shape[1:]))
obs_grp.create_dataset('img_global', data=full_img_global, chunks=(1, *full_img_global.shape[1:]))
data_grp.create_dataset('action', data=full_actions, chunks=(1, *full_actions.shape[1:]))

meta_grp = root.create_group('meta')
meta_grp.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

# 计算并保存归一化统计数据 (非常重要！)
stats = {
    "action_min": full_actions.min(axis=0).tolist(),
    "action_max": full_actions.max(axis=0).tolist()
}
with open(os.path.expanduser("~/workspaces/ur_gz/robot_diffusion_data/stats.json"), "w") as f:
    json.dump(stats, f)

print(f"✅ 合并完成！总帧数: {curr_idx}")
print(f"📦 训练集已保存至: {output_path}")
print(f"📊 统计信息已保存至: stats.json")
