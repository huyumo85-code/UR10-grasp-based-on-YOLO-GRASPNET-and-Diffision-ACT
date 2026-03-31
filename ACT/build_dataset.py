import os
import glob
import zarr
import torch
import numpy as np
from tqdm import tqdm
import shutil

# 自动处理导入路径
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def integrate_zarr_to_lerobot(zarr_dir, repo_id="hym/ur10_act_dataset"):
    # 1. 自动搜索所有的 Zarr 录制文件夹
    zarr_paths = sorted(glob.glob(os.path.join(zarr_dir, "demo_*.zarr")))
    if not zarr_paths:
        print(f"❌ 在 {zarr_dir} 中未找到任何数据文件，请先录制！")
        return

    print(f"🔍 找到 {len(zarr_paths)} 个演示轨迹，准备整合与清洗...")

    # 2. 清理旧数据集缓存，防止与上一次的 Schema 冲突
    local_cache_path = os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_id}")
    if os.path.exists(local_cache_path):
        print(f"🧹 清理旧数据集缓存: {local_cache_path}")
        shutil.rmtree(local_cache_path)

    # 3. 创建全新的 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,  
        robot_type="ur10",
        features={
            "observation.state": {"dtype": "float32", "shape": (7,)},
            "action": {"dtype": "float32", "shape": (7,)},
            "observation.images.cam_high": {"dtype": "video", "shape": (3, 224, 224), "names": ["C", "H", "W"]},
            "observation.images.cam_low": {"dtype": "video", "shape": (3, 224, 224), "names": ["C", "H", "W"]},
        },
        use_videos=True,
    )

    task_label = "pick up the object with UR10"

    total_dropped_frames = 0 # 统计清洗掉的废帧

    # 4. 遍历并合并所有 Zarr 轨迹
    for zarr_path in tqdm(zarr_paths, desc="整合与清洗进度"):
        root = zarr.open(zarr_path, mode='r')
        
        # 读取原始数据
        imgs_hand = root['data/obs/img_hand_eye'][:]  
        imgs_global = root['data/obs/img_global'][:]  
        joints = root['data/action'][:]               
        
        original_frames = joints.shape[0]
        
        # ==========================================
        # 🌟 核心清洗逻辑：寻找机械臂真正开始移动的帧
        # ==========================================
        start_idx = 0
        movement_threshold = 0.005  # 运动阈值 (弧度)，约等于 0.28 度
        first_joint_state = joints[0][:6]  # 仅参考前 6 个手臂关节，忽略夹爪
        
        for i in range(1, original_frames):
            # 计算当前帧与第一帧的最大关节偏差
            max_diff = np.max(np.abs(joints[i][:6] - first_joint_state))
            if max_diff > movement_threshold:
                # 找到真正移动的瞬间！保留前面 2 帧 (0.2s) 作为视觉上下文缓冲
                start_idx = max(0, i - 2)
                break
                
        # 如果清洗后轨迹太短，直接丢弃这个异常轨迹
        if original_frames - start_idx < 15:
            print(f"\n⚠️ 警告: {os.path.basename(zarr_path)} 清洗后有效帧过少，已跳过。")
            continue
            
        dropped = start_idx
        total_dropped_frames += dropped
        
        # 🌟 实施切片裁剪，丢弃发呆帧
        imgs_hand_clean = imgs_hand[start_idx:]
        imgs_global_clean = imgs_global[start_idx:]
        joints_clean = joints[start_idx:]
        num_frames = joints_clean.shape[0]
        
        # 逐帧添加到数据集中
        for i in range(num_frames):
            im_h_t = torch.from_numpy(imgs_hand_clean[i]).permute(2, 0, 1).contiguous()
            im_g_t = torch.from_numpy(imgs_global_clean[i]).permute(2, 0, 1).contiguous()
            state_t = torch.from_numpy(joints_clean[i]).float()
    
            # Action 指向下一帧状态
            idx_next = min(i + 1, num_frames - 1)
            action_t = torch.from_numpy(joints_clean[idx_next]).float()

            frame = {
                "observation.state": state_t,
                "action": action_t,
                "observation.images.cam_high": im_g_t,
                "observation.images.cam_low": im_h_t,
                "task": task_label, 
            }
            dataset.add_frame(frame)
        
        # 保存 Episode
        dataset.save_episode()

    # 5. 整合并打包
    print("\n" + "="*50)
    print(f"🧹 清洗报告: 共砍掉了 {total_dropped_frames} 帧导致模型发呆的冗余数据！")
    print("="*50)
    
    print("\n⏳ 正在进行底层视频压缩与合并 (可能需要几分钟)...")
    print(f"\n✅ 整合完成！共包含 {dataset.num_episodes} 个有效 Episode，{dataset.num_frames} 帧纯净数据。")
    print(f"📍 数据集本地存储位置: {dataset.root}")

if __name__ == "__main__":
    ZARR_DIR = os.path.expanduser("~/workspaces/ur_gz/robot_diffusion_data/zarr_data")
    integrate_zarr_to_lerobot(ZARR_DIR)
