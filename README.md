参考：https://github.com/owenliang/pytorch-diffusion

参考up主：https://www.bilibili.com/video/BV1nP411x7sy/?spm_id_from=333.1391.0.0

# pytorch-diffusion

CIFAR-10的stable diffusion模型复现
使用了pytorch的分布式训练

## 模型效果

引导生成小猫图片

<img width="1500" height="1500" alt="image" src="https://github.com/user-attachments/assets/0f88fc4d-2bb0-4839-ad6b-d1d922153599" />


## 训练主模型

```
torchrun --nproc_per_node=4 train.py
```
## 实现效果
训练损失
<img width="2436" height="1218" alt="image" src="https://github.com/user-attachments/assets/5113a5f6-71b5-448a-ab41-d42a2888a385" />


## 资源占用（为了快用了四张卡， 有点资源浪费了）
GPU显存利用率
<img width="1161" height="308" alt="image" src="https://github.com/user-attachments/assets/3ec8d5ad-9fa6-43de-bbb2-90e35e7da5ee" />

GPU利用率
<img width="1161" height="308" alt="image" src="https://github.com/user-attachments/assets/1cb62d90-f403-416a-a590-91f2bc91f745" />

GPU访问内存所耗时间（%）
<img width="1161" height="308" alt="image" src="https://github.com/user-attachments/assets/58313b9e-0f07-442e-af68-8ddfc05fb688" />

GPU温度
<img width="1161" height="308" alt="image" src="https://github.com/user-attachments/assets/b8215388-c0eb-4877-b820-b059a9b83d86" />

CPU利用率
<img width="1161" height="333" alt="image" src="https://github.com/user-attachments/assets/62ea529c-0af7-4677-b16f-49a801dd34ce" />

