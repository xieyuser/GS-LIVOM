import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

fp = sys.argv[1]
if fp.endswith('.npy'):
    depth_data = np.load(sys.argv[1])
elif fp.endswith(('.png', '.jpg')):
    depth_data = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
else:
    raise ValueError("Invalid file format. Only .npy and .png are supported.")
# 显示数据
plt.imshow(depth_data, cmap='viridis')  # 你可以选择不同的 colormap
plt.colorbar()  # 显示颜色条
plt.title('Depth Map')
plt.axis('off')  # 不显示轴
plt.show()