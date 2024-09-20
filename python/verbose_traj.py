import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data(filename):
    # 读取文本文件
    with open(filename, 'r') as file:
        data_lines = file.readlines()

    # 解析数据
    data = [line.strip().split() for line in data_lines]

    timestamps = [float(row[0]) for row in data]
    quaternions = np.array([[float(row[4]), float(row[5]), float(row[6]), float(row[7])] for row in data])
    positions = np.array([[float(row[1]), float(row[2]), float(row[3])] for row in data])

    return timestamps, quaternions, positions

def calculate_trajectory_length(positions):
    # 计算相邻位置之间的差值
    differences = np.diff(positions, axis=0)
    
    # 计算每一段线段的长度
    segment_lengths = np.linalg.norm(differences, axis=1)
    
    # 返回总长度
    return np.sum(segment_lengths)

def plot_camera_trajectory(positions):
    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color='orange', linewidth=5)  # 设置线条颜色为蓝色，宽度为3

    # 不显示网格
    ax.grid(False)

    # 隐藏所有轴
    ax.axis('off')

    ax.set_aspect('equal')
    # 显示图像
    plt.show()


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    _, _, positions = read_data(filename)
    plot_camera_trajectory(positions)

     # 计算并打印轨迹长度
    trajectory_length = calculate_trajectory_length(positions)
    print(f"轨迹长度: {trajectory_length:.2f}")