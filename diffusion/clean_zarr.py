import zarr
import numpy as np
import os
import shutil
import json

src = os.path.expanduser("~/workspaces/ur_gz/robot_diffusion_data/train_dataset.zarr")
dst = os.path.expanduser("~/workspaces/ur_gz/robot_diffusion_data/train_dataset_clean.zarr")
stats_path = os.path.expanduser("~/workspaces/ur_gz/robot_diffusion_data/stats.json")

if os.path.exists(dst):
    shutil.rmtree(dst)

print("🔍 正在启动【智能清洗】，精准狙击半空发呆死循环...")
z_in = zarr.open(src, mode='r')
act_in = z_in['data/action'][:]
img_h_in = z_in['data/obs/img_hand_eye'][:]
img_g_in = z_in['data/obs/img_global'][:]
ends_in = z_in['meta/episode_ends'][:]

all_h, all_g, all_act, new_ends = [], [], [], []
curr_len = 0
start = 0

for end in ends_in:
    ep_a = act_in[start:end]
    ep_h = img_h_in[start:end]
    ep_g = img_g_in[start:end]

    keep = []
    stationary_counter = 0
    for i in range(len(ep_a)):
        if i == 0:
            keep.append(i)
            continue

        # 提取手臂动作差值和当前的夹爪指令
        diff_arm = np.max(np.abs(ep_a[i][:6] - ep_a[i-1][:6]))
        grip_val = ep_a[i][6]

        # 核心逻辑：手臂没动且夹爪张开 -> 发呆帧，删除
        # grip_val >= 0.3 对应夹爪约 70% 闭合（实际最大值 0.419），判定为抓取阶段
        if diff_arm < 0.0015 and grip_val < 0.3:
            stationary_counter += 1
        else:
            stationary_counter = 0

        # 1 帧缓冲防抽搐；抓取阶段（grip >= 0.3）无条件保留所有静止帧
        if stationary_counter <= 1 or grip_val >= 0.3:
            keep.append(i)

    all_h.append(ep_h[keep])
    all_g.append(ep_g[keep])
    all_act.append(ep_a[keep])
    curr_len += len(keep)
    new_ends.append(curr_len)
    start = end

full_h = np.concatenate(all_h, axis=0)
full_g = np.concatenate(all_g, axis=0)
full_act = np.concatenate(all_act, axis=0)

print(f"✅ 智能清洗完成！从 {len(act_in)} 帧中剔除了 {len(act_in) - len(full_act)} 帧废数据！")

print("📦 正在写入全新的纯净 Zarr...")
root = zarr.open(dst, mode='w')
data_grp = root.create_group('data')
obs_grp = data_grp.create_group('obs')

compressor = zarr.Blosc(cname='zstd', clevel=3)
obs_grp.create_dataset('img_hand_eye', data=full_h, chunks=(100, 224, 224, 3), compressor=compressor)
obs_grp.create_dataset('img_global', data=full_g, chunks=(100, 224, 224, 3), compressor=compressor)
data_grp.create_dataset('action', data=full_act, chunks=(1000, 7))

meta_grp = root.create_group('meta')
meta_grp.create_dataset('episode_ends', data=np.array(new_ends, dtype=np.int64))

print("📊 重新计算动作边界 (stats.json)...")
stats = {
    "action_min": full_act.min(axis=0).tolist(),
    "action_max": full_act.max(axis=0).tolist()
}
with open(stats_path, "w") as f:
    json.dump(stats, f)

print(f"✅ 清洗完毕！总帧数: {len(act_in)} → {len(full_act)}，删除 {len(act_in) - len(full_act)} 帧")

# 用清洁版覆盖原始数据集
shutil.rmtree(src)
os.rename(dst, src)
print("🚀 原始数据集已被纯净版覆盖，可以直接重新训练！")
