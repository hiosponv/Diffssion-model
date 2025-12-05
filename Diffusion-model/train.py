from config import *
from torch.utils.data import DataLoader
from dataset import build_dataset
from unet import UNet
from diffusion import forward_diffusion
import torch 
from torch import nn 
import os 
# from torch.utils.tensorboard import SummaryWriter
import swanlab
import random
from dist_utils import*


local_rank, rank, world_size = setup_ddp()
device = torch.device(f"cuda:{local_rank}")

# 创建一个SwanLab项目,初始化 SwanLab 日志（仅 rank 0 执行一次）
if rank == 0:
    swanlab.init(
        # 设置项目名
        project="diffusion-CIFAR-10",
        
        # 设置超参数
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "UNet",
            "dataset": "CIFAR-10",
            "epochs": EPOCHES
        }
    )

# 创建分布式 DataLoader
train_dataset = build_dataset(rank,True)
val_dataset = build_dataset(rank, False)  # 验证集不做 rank 分配
dataloader, sampler = create_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        world_size=world_size,
        rank=rank,
        shuffle=True
    )
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
try:
    model=torch.load('model.pt', weights_only=False)
except:
    model=UNet(IMG_CHANNELS)   # 噪音预测模型, 彩色图片三通道

# 模型 DDP 化
model = wrap_ddp_model(model=model, local_rank=local_rank)


optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE) # 优化器
loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

# writer = SummaryWriter()
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    for batch_x, batch_cls in val_loader:
        batch_x = batch_x.to(device)*2-1
        batch_cls = batch_cls.to(device)
        batch_t = torch.randint(0, T, (batch_x.size(0),)).to(device)
        batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)
        batch_predict_t = model(batch_x_t, batch_t, batch_cls)
        loss = loss_fn(batch_predict_t, batch_noise_t)
        total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(val_loader.dataset)
    model.train()
    return avg_loss

def main():
    model.train()
    # n_iter=0
    for epoch in range(EPOCHES):
        # 每个 epoch 让 sampler 随机种子保持一致（官方要求）
        sampler.set_epoch(epoch)
        last_loss=0
        for batch_x,batch_cls in dataloader:
            # 图像的像素范围转换到[-1,1],和高斯分布对应
            batch_x=batch_x.to(device)*2-1
            # 引导分类ID
            batch_cls=batch_cls.to(device)
            # 为每张图片生成随机t时刻
            batch_t=torch.randint(0,T,(batch_x.size(0),)).to(device)
            # 生成t时刻的加噪图片和对应噪音
            batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)
            # 模型预测t时刻的噪音
            batch_predict_t=model(batch_x_t,batch_t,batch_cls)
            # 求损失
            loss=loss_fn(batch_predict_t,batch_noise_t)
            # 优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss=loss.item()
            # writer.add_scalar('Loss/train', last_loss, n_iter)
            # rank0 才记录日志
            if rank == 0:
                swanlab.log({"loss": loss})
        if rank == 0:
            val_loss = validate(model, val_dataloader, device)
            print(f'epoch: {epoch}, train_loss={last_loss}, val_loss={val_loss}')
            swanlab.log({"val_loss": val_loss})

            # n_iter+=1
        # 每个 epoch 保存模型（仅 rank0）
        if rank ==0:
            model_save(model,'model.pt.tmp', rank)
            os.replace('model.pt.tmp','model.pt')

    if rank == 0:    
        swanlab.finish()
    # 关闭通信
    cleanup_ddp()

if __name__=='__main__':
    main()