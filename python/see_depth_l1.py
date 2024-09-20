import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import torch

# 读取文件
fp = sys.argv[1]
if fp.endswith('.npy'):
    depth_data = np.load(fp)
elif fp.endswith(('.png', '.jpg')):
    depth_data = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
else:
    raise ValueError("Invalid file format. Only .npy and .png are supported.")


def inv_depth(depth, epsilon=0.01):
    """
    计算深度图的逆深度图。
    
    参数:
    - depth: 输入的深度图数组 (形状为 [H, W] 或 [C, H, W])
    - epsilon: 防止除以零的小数值
    
    返回:
    - inverse_depth: 逆深度图数组
    """
    # 创建掩码，标记深度值小于等于 epsilon 的位置
    mask = depth <= epsilon
    
    # 对深度值进行裁剪，防止除以零
    depth_clamped = np.where(mask, epsilon, depth)
    
    # 计算逆深度
    inverse_depth = 1.0 / depth_clamped
    
    # 使用掩码将小于等于 epsilon 的位置设为 0
    inverse_depth[mask] = 0
    
    return inverse_depth


# 获取图像尺寸
h, w = depth_data.shape[:2]

# 分割图像
src = inv_depth(depth_data[:, :w//2])
dst = inv_depth(depth_data[:, w//2:])
# src = depth_data[:, :w//2]
# dst = depth_data[:, w//2:]
print(f"src shape: {src.shape}, dst shape: {dst.shape}")

# 将 NumPy 数组转换为 PyTorch 张量
src_tensor = torch.from_numpy(src).float()
dst_tensor = torch.from_numpy(dst).float()

# 计算每个像素的 L1 损失
pixel_wise_loss = torch.abs(src_tensor - dst_tensor).numpy()
print(pixel_wise_loss.mean())

# 可视化损失
plt.figure(figsize=(12, 5))

# 显示每个像素的 L1 损失
plt.imshow(pixel_wise_loss, cmap='viridis')
plt.colorbar()
plt.title('Pixel-wise L1 Loss')
plt.axis('off')

plt.tight_layout()
plt.show()