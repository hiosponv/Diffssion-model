参考：https://github.com/owenliang/pytorch-diffusion
参考up主：https://www.bilibili.com/video/BV1nP411x7sy/?spm_id_from=333.1391.0.0

# pytorch-diffusion

CIFAR-10的stable diffusion模型复现
使用了pytorch的分布式训练

## 模型效果

引导生成小猫图片

![alt text](sample_9_denoise.png)

## 训练主模型

```
torchrun --nproc_per_node=4 train.py
```
## 实现效果
训练损失
![alt text](image.png)

## 资源占用（为了快用了四张卡， 有点资源浪费了）
GPU显存利用率
![alt text](image-1.png)
GPU利用率
![alt text](image-2.png)
GPU访问内存所耗时间（%）
![alt text](image-3.png)
GPU温度
![alt text](image-4.png)
CPU利用率
![alt text](image-5.png)