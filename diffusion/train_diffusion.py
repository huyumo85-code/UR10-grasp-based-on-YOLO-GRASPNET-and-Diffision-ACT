import os
import zarr
import json
import torch
import torch.nn as nn
import numpy as np
import sys
from torch.utils.data import DataLoader, Dataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from torchvision import transforms, models

# ================= 🌟 核心超参数 🌟 =================
OBS_HORIZON = 2       # 观察过去 2 帧
ACTION_HORIZON = 16   # 一次性预测未来 16 步的动作轨迹
# ====================================================

# --- 1. 支持序列提取的数据集 ---
class RobotDataset(Dataset):
    def __init__(self, zarr_path, stats_path):
        self.zarr_path = zarr_path
        self.stats_path = stats_path
        
        # 🌟 多进程安全修复：仅在主进程读取元数据
        temp_root = zarr.open(zarr_path, mode='r')
        self.length = temp_root['data/action'].shape[0]
        self.episode_ends = temp_root['meta/episode_ends'][:]
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
            
        self.root = None # 留给 worker 进程初始化

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize((224, 224), antialias=True)

    def _init_zarr(self):
        if self.root is None:
            self.root = zarr.open(self.zarr_path, mode='r')
            self.img_hand = self.root['data/obs/img_hand_eye']
            self.img_global = self.root['data/obs/img_global']
            self.actions = self.root['data/action']

    def __len__(self):
        return self.length

    def _get_episode_boundaries(self, idx):
        ep_idx = np.searchsorted(self.episode_ends, idx, side='right')
        ep_start = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
        ep_end = self.episode_ends[ep_idx]
        return ep_start, ep_end

    def __getitem__(self, idx):
        self._init_zarr() # 延迟初始化
        ep_start, ep_end = self._get_episode_boundaries(idx)

        obs_indices = np.arange(idx - OBS_HORIZON + 1, idx + 1)
        obs_indices = np.clip(obs_indices, ep_start, ep_end - 1)

        action_indices = np.arange(idx, idx + ACTION_HORIZON)
        action_indices = np.clip(action_indices, ep_start, ep_end - 1)

        a_min = np.array(self.stats['action_min'])
        a_max = np.array(self.stats['action_max'])

        action_seq = self.actions.oindex[action_indices]
        action_seq = (action_seq - a_min) / (a_max - a_min + 1e-6) * 2 - 1

        state_seq = self.actions.oindex[obs_indices]
        state_seq = (state_seq - a_min) / (a_max - a_min + 1e-6) * 2 - 1

        def prep_img_seq(img_array, indices):
            min_idx, max_idx = int(min(indices)), int(max(indices))
            chunk = img_array[min_idx : max_idx + 1] 
            
            imgs = []
            for i in indices:
                rel_idx = i - min_idx
                img = torch.from_numpy(chunk[rel_idx]).permute(2, 0, 1).float() / 255.0
                img = self.resize(img)
                imgs.append(self.normalize(img))
            return torch.stack(imgs) 

        return {
            'img_hand': prep_img_seq(self.img_hand, obs_indices),
            'img_global': prep_img_seq(self.img_global, obs_indices),
            'current_state': torch.from_numpy(state_seq).float(),
            'action': torch.from_numpy(action_seq).float()
        }

# --- 2. 支持序列输入的扩散网络 ---
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_hand = models.resnet18(weights='IMAGENET1K_V1')
        self.vision_global = models.resnet18(weights='IMAGENET1K_V1')
        self.vision_hand.fc = nn.Identity()
        self.vision_global.fc = nn.Identity()
        
        self.state_encoder = nn.Linear(7, 64) 

        # 🌟 修复：为 Timestep 添加专门的 Embedding 层
        self.time_emb_dim = 32
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        cam_feat_dim = 512 * 2 * OBS_HORIZON   
        state_feat_dim = 64 * OBS_HORIZON      
        action_feat_dim = 7 * ACTION_HORIZON   
        in_dim = cam_feat_dim + state_feat_dim + action_feat_dim + self.time_emb_dim 

        self.noise_pred = nn.Sequential(
            nn.Linear(in_dim, 1024), 
            nn.LayerNorm(1024), 
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, action_feat_dim) 
        )

    def forward(self, img_h, img_g, current_state, action, timestep):
        B, T, C, H, W = img_h.shape
        
        img_h_flat = img_h.view(B * T, C, H, W)
        img_g_flat = img_g.view(B * T, C, H, W)

        f_h = self.vision_hand(img_h_flat).view(B, -1) 
        f_g = self.vision_global(img_g_flat).view(B, -1)
        f_s = self.state_encoder(current_state).view(B, -1) 
        
        obs_feat = torch.cat([f_h, f_g, f_s], dim=-1) 
        action_flat = action.view(B, -1) 
        
        # 🌟 修复：Timestep 必须经过 Embedding 处理
        t_normalized = (timestep.float() / 100.0).unsqueeze(-1)
        t_emb = self.time_embed(t_normalized)
        
        h = torch.cat([obs_feat, action_flat, t_emb], dim=-1)
        out_flat = self.noise_pred(h) 
        
        return out_flat.view(B, -1, 7)

# --- 3. 训练循环 ---
def train():
    device = torch.device('cuda')
    dataset = RobotDataset(
        os.path.expanduser('~/workspaces/ur_gz/robot_diffusion_data/train_dataset.zarr'),
        os.path.expanduser('~/workspaces/ur_gz/robot_diffusion_data/stats.json')
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    model = DiffusionModel().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,  
        num_training_steps=len(loader) * 201 
    )    

    print("🚀 带着全新的架构和 Loss，从零开始训练...")
    
    for epoch in range(201):
        total_loss = 0
        for batch in loader:
            img_h = batch['img_hand'].to(device)
            img_g = batch['img_global'].to(device)
            current_state = batch['current_state'].to(device)
            action = batch['action'].to(device) 
            
            noise = torch.randn(action.shape).to(device)
            timesteps = torch.randint(0, 100, (action.shape[0],), device=device).long()
            
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
            
            # 注意：传入原生的整数 timestep，模型内部会做归一化
            noise_pred = model(img_h, img_g, current_state, noisy_action, timesteps)

            # 🌟 你的终极手腕特训 Loss
            mse_base = nn.functional.mse_loss(noise_pred[:, :, :3], noise[:, :, :3])
            mse_wrist = nn.functional.mse_loss(noise_pred[:, :, 3:6], noise[:, :, 3:6])
            mse_gripper = nn.functional.mse_loss(noise_pred[:, :, 6:], noise[:, :, 6:])
            
            wrist_weight = 10.0
            gripper_weight = 5.0
            loss = mse_base + (wrist_weight * mse_wrist) + (gripper_weight * mse_gripper)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            
        # 🌟 修复后的 Print 语句
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.6f} (Base: {mse_base.item():.4f}, Wrist: {mse_wrist.item():.4f}, Grip: {mse_gripper.item():.4f})")
            
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'diffusion_model_epoch_{epoch}.pth')

if __name__ == '__main__':
    train()
