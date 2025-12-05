import torch 
from config import *
from diffusion import *
import matplotlib.pyplot as plt 
from dataset import tensor_to_pil
from lora import LoraLayer
from torch import nn 
from get_classes import get_batch_cls


def backward_denoise(model, batch_x_t, batch_cls):
    steps=[batch_x_t,]

    global alphas,alphas_cumprod,variance

    model = model.to(DEVICE)
    batch_x_t = batch_x_t.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    variance = variance.to(DEVICE)
    batch_cls = batch_cls.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        for t in range(T-1, -1, -1):
            batch_t = torch.full((batch_x_t.size(0),), t).to(DEVICE)
            batch_predict_noise_t = model(batch_x_t, batch_t, batch_cls)
            shape = (batch_x_t.size(0), 1, 1, 1)
            batch_mean_t = 1 / torch.sqrt(alphas[batch_t].view(*shape)) * (
                batch_x_t - (1 - alphas[batch_t].view(*shape)) / 
                torch.sqrt(1 - alphas_cumprod[batch_t].view(*shape)) * batch_predict_noise_t
            )
            if t != 0:
                batch_x_t = batch_mean_t + torch.randn_like(batch_x_t) * torch.sqrt(variance[batch_t].view(*shape))
            else:
                batch_x_t = batch_mean_t
            batch_x_t = torch.clamp(batch_x_t, -1.0, 1.0).detach()
            steps.append(batch_x_t)
    return steps 

def main():
    # 加载模型
    model = torch.load('model.pt', weights_only=False)
    model = model.to(DEVICE)

    # 生成噪音图
    batch_size = 10
    batch_x_t = torch.randn(size=(batch_size, 3, IMG_SIZE, IMG_SIZE))  # 例如 (10,1,48,48)

    # 示例 1：全部生成 cat
    batch_cls = get_batch_cls("cat", batch_size)
    # 示例 2：生成多个类别循环填充
    # batch_cls = get_batch_cls(["cat","dog","frog"], batch_size)

    # 逐步去噪得到原图
    steps = backward_denoise(model, batch_x_t, batch_cls)

    # 绘制还原过程
    num_imgs = 20
    plt.figure(figsize=(15,15))
    for b in range(batch_size):
        for i in range(num_imgs):
            idx = int(T / num_imgs) * (i+1)
            final_img = (steps[idx][b].to('cpu') + 1) / 2  # 归一化到 [0,1]
            final_img = tensor_to_pil(final_img)
            plt.subplot(batch_size, num_imgs, b*num_imgs + i + 1)
            plt.imshow(final_img)
    plt.show()


if __name__=='__main__':
    main()