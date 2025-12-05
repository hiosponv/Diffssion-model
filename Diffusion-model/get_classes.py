import torch

# CIFAR-10 类别到索引的映射
CIFAR10_CLASSES = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

def get_batch_cls(class_names, batch_size):
    """
    class_names: str 或 list[str]，要生成的类别名
    batch_size: int,生成的 batch 大小
    返回: batch_cls Tensor
    """
    if isinstance(class_names, str):
        # 如果是单个类别，生成全 batch
        class_idx = CIFAR10_CLASSES[class_names]
        batch_cls = torch.full((batch_size,), class_idx, dtype=torch.long)
    elif isinstance(class_names, list):
        # 如果是多个类别，循环填充
        class_indices = [CIFAR10_CLASSES[name] for name in class_names]
        batch_cls = torch.tensor(
            [class_indices[i % len(class_indices)] for i in range(batch_size)],
            dtype=torch.long
        )
    else:
        raise ValueError("class_names 必须是 str 或 list[str]")
    return batch_cls