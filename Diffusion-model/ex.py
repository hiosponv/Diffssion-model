# import torch
# from torch import nn
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List
# import torchvision 
# from torchvision import transforms 
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.distributed import DistributedSampler
# import os
# import time
# import math

# # DDP 相关的导入
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.cuda.amp import autocast, GradScaler

# # =========================================================
# # 0. Constants and Global Configuration
# # =========================================================
# IMG_SIZE = 32 # CIFAR-10 图像大小
# T = 1000 
# # 全局 BATCH_SIZE (将被分配给所有 GPU)
# BATCH_SIZE_GLOBAL = 400 
# EPOCHES = 500 
# LEARNING_RATE = 1e-3
# GRADIENT_CLIP_VALUE = 1.0 
# # 定义全局设备，主进程使用 CUDA:0 或 CPU
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# IMG_CHANNELS = 3 # CIFAR-10 是彩色图像 (RGB)

# # UNet 结构配置
# # 当前配置 (默认): [64, 128, 256, 512, 1024]
# # 如果需要加大模型 (增加宽度), 可以修改为例如: [128, 256, 512, 1024, 1024]
# UNET_CHANNEL_LIST = [64, 128, 256, 512, 1024] 


# # =========================================================
# # 1. Transforms and Data Loading 
# # =========================================================

# # PLT图像转向量
# pil_to_tensor = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)), 
#     torchvision.transforms.ToTensor()]
# )

# # tensor转PIL图像
# tensor_to_pil = torchvision.transforms.Compose(
#     [torchvision.transforms.Lambda(lambda t: t * 255),
#      torchvision.transforms.Lambda(lambda t: t.type(torch.uint8)), torchvision.transforms.ToPILImage()
#     ]
# )

# # 加载 CIFAR-10 数据集
# train_dataset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=pil_to_tensor)


# # =========================================================
# # 2. Time Embedding and Diffusion Schedule 
# # =========================================================

# class TimePositionEmbedding(nn.Module):
#     def __init__(self, emb_size):
#         super().__init__()
#         self.half_emb_size = emb_size // 2
#          # 使用 T=10000 的经典公式来计算频率
#         half_emb = torch.exp(torch.arange(self.half_emb_size) * (-1 * math.log(10000) / (self.half_emb_size - 1)))
#         self.register_buffer('half_emb', half_emb)

#     def forward(self, t): # t:(batch_size,)
#         t = t.view(t.size(0), 1)
#         half_emb = self.half_emb.unsqueeze(0).expand(t.size(0), self.half_emb_size)
#         half_emb_t = half_emb * t
#         embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1)
#         return embs_t

# # 前向diffusion计算参数 (T,)
# betas = torch.linspace(0.0001, 0.02, T)
# alphas = 1 - betas
# alphas_cumprod = torch.cumprod(alphas, dim=-1)
# alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1)
# variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

# # 前向加噪 (Forward Diffusion)
# def forward(batch_x, batch_t, device): 
#     # 确保 diffusion 参数在正确的设备上
#     local_alphas_cumprod = alphas_cumprod.to(device=device)
    
#     batch_noise_t = torch.randn_like(batch_x) # 为每张图片生成第t步的高斯噪音   (batch,channel,width,height)
#     # 取出和时间步一致的aplhas_cumprod，并重塑以进行广播
#     batch_alphas_cumprod = local_alphas_cumprod[batch_t].view(batch_x.size(0), 1, 1, 1) 
#     batch_x_t = torch.sqrt(batch_alphas_cumprod) * batch_x + torch.sqrt(1 - batch_alphas_cumprod) * batch_noise_t # 基于公式直接生成第t步加噪后图片
#     return batch_x_t, batch_noise_t

# # =========================================================
# # 3. UNet Model Architecture 
# # =========================================================

# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, time_emb_size):
#         super().__init__()

#         self.seq1 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )

#         self.time_emb_linear = nn.Linear(time_emb_size, out_channel)
#         self.relu = nn.ReLU()

#         self.seq2 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )


#     def forward(self, x, t_emb):
#         x = self.seq1(x) 
#         # t_emb: (batch_size, out_channel, 1, 1)
#         t_emb = self.relu(self.time_emb_linear(t_emb)).view(x.size(0), x.size(1), 1, 1) 
#         output = self.seq2(x + t_emb) # 时间信息注入
#         return output

# class UNet(nn.Module):
#     # 现在 channels 列表必须在初始化时传入，以便使用 UNET_CHANNEL_LIST 全局变量
#     def __init__(self, img_channel, channels: List[int], time_emb_size: int=256):
#         super().__init__()

#         full_channels = [img_channel] + channels

#         # time转embedding
#         self.time_emb = nn.Sequential(
#             TimePositionEmbedding(emb_size=time_emb_size),
#             nn.Linear(time_emb_size, time_emb_size),
#             nn.ReLU()
#          )

#         # Encoder stages
#         self.enc_convs = nn.ModuleList()
#         for i in range(len(full_channels) - 1):
#          self.enc_convs.append(ConvBlock(full_channels[i], full_channels[i+1], time_emb_size))
#         # MaxPools (比 ConvBlocks 少一个)
#         self.maxpools = nn.ModuleList()
#         for i in range(len(full_channels) - 2):
#             self.maxpools.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

#         # Decoder ConvTranspose layers (用于上采样)
#         self.deconvs=nn.ModuleList()
#         for i in range(len(full_channels) - 2):
#             self.deconvs.append(nn.ConvTranspose2d(full_channels[-i-1],full_channels[-i-2],kernel_size=2,stride=2))

#         # Decoder ConvBlocks
#         self.dec_convs = nn.ModuleList()
#         for i in range(len(full_channels) - 2):
#             self.dec_convs.append(ConvBlock(full_channels[-i-1], full_channels[-i-2], time_emb_size))

#         # 还原通道数 (输出必须是 img_channel，即 3)
#         self.output = nn.Conv2d(full_channels[1], img_channel, kernel_size=1, stride=1, padding=0)
    
#     # 假设 UNet 的 forward 签名与您原训练代码一致，接受 x, t, batch_cls
#     def forward(self, x, t, batch_cls=None): 
#         t_emb = self.time_emb(t)

#         # encoder阶段
#         residual = []
#         num_downsamples = len(self.maxpools)
#         for i, conv in enumerate(self.enc_convs):
#             x = conv(x, t_emb)
#             if i < num_downsamples: # 最后一个 conv 后不缩小
#                 residual.append(x)
#                 x = self.maxpools[i](x)

#         # decoder阶段
#         for i, deconv in enumerate(self.deconvs):
#             x = deconv(x)
#             residual_x = residual.pop(-1)
#             x = self.dec_convs[i](torch.cat((x, residual_x), dim=1), t_emb)
#             return self.output(x)

# # =========================================================
# # 4. Backward Denoise (Sampling) Function
# # =========================================================

# def backward_denoise(model, batch_x_t, device):
#     steps = [batch_x_t.clone().detach().cpu(), ] 
    
#     # 确保扩散参数在正确的设备上
#     local_alphas = alphas.to(device)
#     local_alphas_cumprod = alphas_cumprod.to(device)
#     local_variance = variance.to(device)

#     batch_x_t = batch_x_t.to(device)

#     with torch.no_grad():
#         for t in range(T-1, -1, -1):
#             batch_t = torch.full((batch_x_t.size(0),), t).to(device)
#             # 注意：采样时默认不传递 batch_cls (无条件生成)
#             batch_predict_noise_t = model(batch_x_t, batch_t)
#             # DDPM 逆向均值和方差计算
#             shape = (batch_x_t.size(0), 1, 1, 1)
            
#             # 从全局张量中获取当前时间步 t 的参数，并重塑
#             alpha_t = local_alphas[batch_t].view(*shape)
#             alpha_cumprod_t = local_alphas_cumprod[batch_t].view(*shape)
#             variance_t = local_variance[batch_t].view(*shape)

#             # DDPM 均值公式 (tilde_mu)
#             batch_mean_t=1/torch.sqrt(alpha_t) * \
#                 (
#                     batch_x_t-
#                      ((1-alpha_t)/torch.sqrt(1-alpha_cumprod_t)) * batch_predict_noise_t
#                 ) 
#             if t != 0:
#                 # 添加高斯噪声
#                 batch_x_t = batch_mean_t + \
#                  torch.randn_like(batch_x_t) * torch.sqrt(variance_t)
#             else:
#                 batch_x_t = batch_mean_t
#             batch_x_t = torch.clamp(batch_x_t, -1.0, 1.0).detach()
#             steps.append(batch_x_t.cpu())

#     return steps


# # =========================================================
# # 5. Main DDP Worker Function (Training)
# # =========================================================

# def setup(rank, world_size):
#     """初始化 DDP 进程组"""
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355' 
#     dist.init_process_group("nccl", rank=rank, world_size=world_size) 

# def cleanup():
#     """销毁 DDP 进程组"""
#     dist.destroy_process_group()

# def main_worker(rank, world_size, train_dataset):
#     """ DDP 训练的主要执行函数。"""
#     setup(rank, world_size)
    
#     device = torch.device(f"cuda:{rank}")
#     print(f"Rank {rank} initializing on device {device}")
    
#     # ------------------- 1. 模型初始化与 DDP 包装 -------------------
#     # 使用全局常量 UNET_CHANNEL_LIST 初始化 UNet 
#     model = UNet(img_channel=IMG_CHANNELS, channels=UNET_CHANNEL_LIST).to(device)
    
#     # 尝试加载模型状态 (仅 Rank 0 进行文件操作和加载)
#     if rank == 0:
#         try:
#             state_dict = torch.load('model_cifar10.pt', map_location=device)
#             # 移除 DDP 包装时可能存在的 'module.' 前缀
#             new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#             model.load_state_dict(new_state_dict)
#             print(f"Rank {rank}: Model loaded successfully from model_cifar10.pt.")
#         except Exception as e:
#             print(f"Rank {rank}: No saved model found or load failed ({e}). Initializing new UNet.")

#     # 将模型包装为 DDP
#     model = DDP(model, device_ids=[rank])
    
#     # ------------------- 2. DataLoader 和 Sampler -------------------
#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     batch_size_per_gpu = BATCH_SIZE_GLOBAL // world_size
    
#     dataload = DataLoader(
#         train_dataset, 
#         batch_size=batch_size_per_gpu, 
#         num_workers=4, 
#         persistent_workers=True, 
#         sampler=train_sampler
#     )

#     # ------------------- 3. 优化器、损失函数与 AMP 缩放器 -------------------
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     loss_fnn = nn.L1Loss()
#     scaler = GradScaler() 

#     # ------------------- 4. 训练循环 -------------------
#     model.train()
#     for epoch in range(EPOCHES):
#         train_sampler.set_epoch(epoch)
#         last_loss = 0
        
#         for batch_x, batch_cls in dataload:
#             # 数据发送到当前进程的 GPU
#             batch_x = batch_x.to(device) * 2 - 1 # 归一化到 [-1, 1]
#             batch_cls = batch_cls.to(device)
#             batch_t = torch.randint(0, T, (batch_x.size(0),)).to(device)
            
#             with autocast():
#                 batch_x_t, batch_noise_t = forward(batch_x, batch_t, device)
#                 batch_predict_t = model(batch_x_t, batch_t, batch_cls)
#                 loss = loss_fnn(batch_noise_t, batch_predict_t)

#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer) 
#             torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
#             scaler.step(optimizer)
#             scaler.update()
            
#             last_loss = loss.item()
            
#         # ------------------- 5. 日志与保存 (仅在 Rank 0 进行) -------------------
#         if rank == 0:
#             print('Epoch:{} Loss={:.6f}'.format(epoch, last_loss))
#             # 保存到新的文件名，避免与 MNIST 模型混淆
#             torch.save(model.module.state_dict(), 'model_cifar10.pt.tmp')
#             os.replace('model_cifar10.pt.tmp', 'model_cifar10.pt')
            
#     cleanup()

# # =========================================================
# # 6. Sampling Function (Runs after DDP Training)
# # =========================================================

# def run_sampling():
#     """在训练完成后，使用 Rank 0 保存的模型进行采样和可视化。"""
#     print("\n--- Starting Sampling ---")
    
#     # 1. 实例化模型结构 (通道数依然是 3, 使用同样的通道列表)
#     model = UNet(img_channel=IMG_CHANNELS, channels=UNET_CHANNEL_LIST).to(DEVICE)
    
#     # 2. 正确加载 state_dict
#     try:
#         state_dict = torch.load("model_cifar10.pt", map_location=DEVICE)
#         model.load_state_dict(state_dict)
#         print("Model state_dict loaded successfully for sampling.")
#     except Exception as e:
#         print(f"Error loading model state_dict: {e}. Cannot sample without trained model.")
#         return

#     model.eval()
#     batch_size = 10
#     num_imgs = 20
    
#     # 随机生成纯高斯噪声 x_T
#     batch_x_t = torch.randn(size=(batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
    
#     # 执行逆向去噪
#     steps = backward_denoise(model, batch_x_t, DEVICE)
    
#     # ------------------ 绘制还原过程 ------------------
#     if not steps:
#         print("Sampling failed. No steps generated.")
#     else:
#         plt.figure(figsize=(num_imgs * 1.5, batch_size * 1.5))
#         plt.suptitle(f"DDPM CIFAR-10 Denoising Process (T={T} steps)", fontsize=16)

#         for b in range(batch_size):
#             for num in range(0, num_imgs):
#                 # 确保索引是整数
#                 idx = int((T/num_imgs) * (num + 1))
                
#                 # 边界检查
#                 if idx >= len(steps):
#                     idx = len(steps) - 1
                
#                 # 还原像素: 将 [-1, 1] 映射到 [0, 1]
#                 final_img = (steps[idx][b].to('cpu') + 1) / 2
#                 final_img_pil = tensor_to_pil(final_img)
                
#                 plt.subplot(batch_size, num_imgs, b*num_imgs+num+1)
#                 plt.imshow(final_img_pil) # CIFAR-10 是彩色图像，移除 cmap='gray'
#                 plt.title(f"t={T - idx}", fontsize=8) 
#                 plt.axis('off')
                
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()


# def main():
#     """主函数：获取 GPU 数量并启动 DDP 进程"""
#     world_size = torch.cuda.device_count()
#     if world_size < 1:
#         print("未检测到 CUDA 设备。将退回到 CPU 运行，但 DDP 无法启动。")
#         # 如果没有 GPU，直接跳过 DDP 训练，进行采样测试
#         # 为了运行 CIFAR-10，需要先下载数据集
#         _ = torchvision.datasets.CIFAR10(root='.', train=True, download=True)
#         run_sampling() 
#         return
        
#     print(f"检测到 {world_size} 块 GPU。将启动 DDP 训练。")
#     # 使用 mp.spawn 启动多个进程
#     mp.spawn(main_worker,
#              args=(world_size, train_dataset),
#              nprocs=world_size,
#              join=True)

# if __name__ == "__main__":
#     main()
    
#     # 确保 DDP 进程结束后，主进程可以进行采样
#     if torch.cuda.device_count() > 0:
#         run_sampling()