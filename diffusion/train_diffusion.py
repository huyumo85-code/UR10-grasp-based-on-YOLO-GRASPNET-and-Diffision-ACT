import os
import zarr
import json
import warnings
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', message='Detected call of')
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from torchvision import transforms, models

# ================= 核心超参数 =================
OBS_HORIZON    = 2    # 观察过去 2 帧
ACTION_HORIZON = 16   # 一次性预测未来 16 步的动作轨迹
NUM_EPOCHS     = 500
BATCH_SIZE     = 16
LR_BACKBONE    = 1e-5  # ResNet 用小学习率，避免破坏预训练特征
LR_HEAD        = 1e-4  # 噪声预测头用正常学习率
# =============================================


# --- 残差块 ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.block(x)


# --- 1. 数据集（含训练期图像增强）---
class RobotDataset(Dataset):
    def __init__(self, zarr_path, stats_path, augment=False):
        self.zarr_path = zarr_path
        self.augment   = augment

        temp_root = zarr.open(zarr_path, mode='r')
        self.length       = temp_root['data/action'].shape[0]
        self.episode_ends = temp_root['meta/episode_ends'][:]
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)

        self.root = None

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize((224, 224), antialias=True)

        # 训练时的数据增强：随机色彩抖动（应对光照变化）
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

    def _init_zarr(self):
        if self.root is None:
            self.root      = zarr.open(self.zarr_path, mode='r')
            self.img_hand  = self.root['data/obs/img_hand_eye']
            self.img_global = self.root['data/obs/img_global']
            self.actions   = self.root['data/action']

    def __len__(self):
        return self.length

    def _get_episode_boundaries(self, idx):
        ep_idx   = np.searchsorted(self.episode_ends, idx, side='right')
        ep_start = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
        ep_end   = self.episode_ends[ep_idx]
        return ep_start, ep_end

    def __getitem__(self, idx):
        self._init_zarr()
        ep_start, ep_end = self._get_episode_boundaries(idx)

        obs_indices    = np.clip(np.arange(idx - OBS_HORIZON + 1, idx + 1),
                                 ep_start, ep_end - 1)
        action_indices = np.clip(np.arange(idx, idx + ACTION_HORIZON),
                                 ep_start, ep_end - 1)

        a_min = np.array(self.stats['action_min'])
        a_max = np.array(self.stats['action_max'])

        action_seq = self.actions.oindex[action_indices]
        action_seq = (action_seq - a_min) / (a_max - a_min + 1e-6) * 2 - 1

        state_seq = self.actions.oindex[obs_indices]
        state_seq = (state_seq - a_min) / (a_max - a_min + 1e-6) * 2 - 1

        def prep_img_seq(img_array, indices):
            min_idx = int(min(indices))
            max_idx = int(max(indices))
            chunk = img_array[min_idx: max_idx + 1]
            imgs = []
            for i in indices:
                img = torch.from_numpy(
                    chunk[i - min_idx]).permute(2, 0, 1).float() / 255.0
                img = self.resize(img)
                if self.augment:
                    img = self.color_jitter(img)
                imgs.append(self.normalize(img))
            return torch.stack(imgs)

        return {
            'img_hand':      prep_img_seq(self.img_hand,   obs_indices),
            'img_global':    prep_img_seq(self.img_global, obs_indices),
            'current_state': torch.from_numpy(state_seq).float(),
            'action':        torch.from_numpy(action_seq).float()
        }


# --- 2. 扩散网络（全量 ResNet 微调 + 残差噪声预测器）---
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_hand   = models.resnet18(weights='IMAGENET1K_V1')
        self.vision_global = models.resnet18(weights='IMAGENET1K_V1')
        self.vision_hand.fc   = nn.Identity()
        self.vision_global.fc = nn.Identity()
        # 全量解冻，由优化器的分层学习率控制更新幅度

        self.state_encoder = nn.Linear(7, 64)

        self.time_emb_dim = 32
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        cam_feat_dim    = 512 * 2 * OBS_HORIZON
        state_feat_dim  = 64 * OBS_HORIZON
        action_feat_dim = 7 * ACTION_HORIZON
        in_dim = cam_feat_dim + state_feat_dim + action_feat_dim + self.time_emb_dim

        self.noise_pred = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            ResBlock(1024),
            nn.Dropout(0.1),
            ResBlock(1024),
            nn.Dropout(0.1),
            ResBlock(1024),
            nn.Linear(1024, action_feat_dim)
        )

    def forward(self, img_h, img_g, current_state, action, timestep):
        B, T, C, H, W = img_h.shape

        f_h = self.vision_hand(img_h.view(B * T, C, H, W)).view(B, -1)
        f_g = self.vision_global(img_g.view(B * T, C, H, W)).view(B, -1)
        f_s = self.state_encoder(current_state).view(B, -1)

        obs_feat    = torch.cat([f_h, f_g, f_s], dim=-1)
        action_flat = action.view(B, -1)
        t_emb       = self.time_embed((timestep.float() / 100.0).unsqueeze(-1))

        out = self.noise_pred(torch.cat([obs_feat, action_flat, t_emb], dim=-1))
        return out.view(B, -1, 7)


# --- 3. 训练循环 ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")

    dataset = RobotDataset(
        os.path.expanduser('~/workspaces/ur_gz/robot_diffusion_data/train_dataset.zarr'),
        os.path.expanduser('~/workspaces/ur_gz/robot_diffusion_data/stats.json'),
        augment=True
    )
    print(f"📦 数据集总帧数: {len(dataset)}")

    history = {'loss': [], 'base': [], 'wrist': [], 'grip': []}

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)
    model = DiffusionModel().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=100)

    # 分层学习率：ResNet 用小 lr，噪声预测头用大 lr
    backbone_params = (list(model.vision_hand.parameters()) +
                       list(model.vision_global.parameters()))
    head_params     = (list(model.state_encoder.parameters()) +
                       list(model.time_embed.parameters()) +
                       list(model.noise_pred.parameters()))
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ])

    scaler = GradScaler()

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(loader) * NUM_EPOCHS
    )

    print(f"🚀 开始训练，共 {NUM_EPOCHS} 轮，每轮 {len(loader)} 个 batch...")

    for epoch in range(NUM_EPOCHS):
        total_loss = total_base = total_wrist = total_gripper = 0.0

        for batch in loader:
            img_h         = batch['img_hand'].to(device)
            img_g         = batch['img_global'].to(device)
            current_state = batch['current_state'].to(device)
            action        = batch['action'].to(device)

            noise     = torch.randn(action.shape, device=device)
            timesteps = torch.randint(0, 100, (action.shape[0],), device=device).long()

            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

            with autocast():
                noise_pred = model(img_h, img_g, current_state, noisy_action, timesteps)

                mse_base    = nn.functional.mse_loss(noise_pred[:, :, :3],  noise[:, :, :3])
                mse_wrist   = nn.functional.mse_loss(noise_pred[:, :, 3:6], noise[:, :, 3:6])
                mse_gripper = nn.functional.mse_loss(noise_pred[:, :, 6:],  noise[:, :, 6:])

                loss = mse_base + 10.0 * mse_wrist + 20.0 * mse_gripper

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # 梯度裁剪：防止全量微调时梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss    += loss.item()
            total_base    += mse_base.item()
            total_wrist   += mse_wrist.item()
            total_gripper += mse_gripper.item()

        n = len(loader)
        avg_loss    = total_loss    / n
        avg_base    = total_base    / n
        avg_wrist   = total_wrist   / n
        avg_gripper = total_gripper / n

        history['loss'].append(avg_loss)
        history['base'].append(avg_base)
        history['wrist'].append(avg_wrist)
        history['grip'].append(avg_gripper)

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
              f"Loss: {avg_loss:.5f} | "
              f"Base: {avg_base:.4f} | "
              f"Wrist: {avg_wrist:.4f} | "
              f"Grip: {avg_gripper:.4f}")

        save_dir = os.path.dirname(os.path.abspath(__file__))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'diffusion_model_epoch_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(save_dir, f'diffusion_model_epoch_{NUM_EPOCHS - 1}.pth'))
    print("✅ 训练完成！正在保存 loss 曲线...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Loss Curves', fontsize=14)
    ep = range(NUM_EPOCHS)

    axes[0, 0].plot(ep, history['loss'],  'b');    axes[0, 0].set_title('Total Loss');       axes[0, 0].set_xlabel('Epoch')
    axes[0, 1].plot(ep, history['base'],  'g');    axes[0, 1].set_title('Base Loss (J1-J3)'); axes[0, 1].set_xlabel('Epoch')
    axes[1, 0].plot(ep, history['wrist'], 'orange');axes[1, 0].set_title('Wrist Loss (J4-J6)');axes[1, 0].set_xlabel('Epoch')
    axes[1, 1].plot(ep, history['grip'],  'r');    axes[1, 1].set_title('Gripper Loss');      axes[1, 1].set_xlabel('Epoch')

    plt.tight_layout()
    save_path = os.path.expanduser('~/workspaces/ur_gz/diffusion/loss_curve.png')
    plt.savefig(save_path, dpi=150)
    print(f"📊 Loss 曲线已保存至: {save_path}")


if __name__ == '__main__':
    train()
