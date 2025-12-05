import torch 

IMG_SIZE=32   # 图像尺寸
T=1000   # 加噪最大步数
LORA_ALPHA=1    # lora的a权重
LORA_R=8    # lora的秩
UNET_CHANNEL_LIST=[128, 256, 512, 1024, 2048]
IMG_CHANNELS = 3 
LEARNING_RATE = 1e-3
BATCH_SIZE=400
EPOCHES = 500 
GRADIENT_CLIP_VALUE = 1.0 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备