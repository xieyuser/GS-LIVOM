import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from matplotlib.colors import Normalize

plt.rcParams['font.size'] = 12


def hex_to_rgb(hex_color):
    """
    Converts a hexadecimal color to RGB tuple.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))



def plot_ellipsoid(mean, cov, ax, n_std=3.0, has_quiver=False, **kwargs):
    """
    Plots an ellipsoid given the mean and covariance matrix.
    Also plots the shortest principal axis as an arrow.
    """
    U, s, Vt = np.linalg.svd(cov)
    radii = n_std * np.sqrt(s) / 2

    u = np.linspace(0., 2 * np.pi, 100)
    v = np.linspace(0., np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(2 * np.ones_like(u), np.cos(v))

    # Transform the ellipsoid coordinates using the rotation matrix U
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], U) + mean

    # Plot the ellipsoid using wireframe
    ax.plot_wireframe(x, y, z, rstride=1, cstride=4, color=hex_to_rgb('#A3C2A3'))  # Use wireframe with no color

    # Plot the shortest principal axis as an arrow
    arrow_length = 0.6 * n_std * np.sqrt(s.mean())  # Length of the shortest arrow
    shortest_axis_index = np.argmin(s)
    if not has_quiver:
        ax.quiver(mean[0], mean[1], mean[2],
                U[:, shortest_axis_index][0] * arrow_length,
                U[:, shortest_axis_index][1] * arrow_length,
                U[:, shortest_axis_index][2] * arrow_length,
                color='r', arrow_length_ratio=0.1)
    else:
        ax.quiver(mean[0], mean[1], mean[2],
                U[:, shortest_axis_index][0] * arrow_length,
                U[:, shortest_axis_index][1] * arrow_length,
                U[:, shortest_axis_index][2] * arrow_length,
                color='r', arrow_length_ratio=0.1, label='Normals')

def fit_ellipsoids(points, num_gp_side=9, neighbour_size=3):
    means = []
    covariance_matrices = []
    grid_size = num_gp_side // neighbour_size

    for i in range(grid_size):
        for j in range(grid_size):
            block_points = []
            for di in range(neighbour_size):
                for dj in range(neighbour_size):
                    index = (i * neighbour_size + di) * num_gp_side + (j * neighbour_size + dj)
                    block_points.append(points[index])

            coordinates = np.array([(p[0], p[1], p[2]) for p in block_points])
            weights = np.array([1.0  for p in block_points])

            weighted_mean = np.sum(coordinates * weights[:, None], axis=0) / np.sum(weights)

            centered = coordinates - weighted_mean
            weighted_covariance = np.dot((centered * weights[:, None]).T, centered) / np.sum(weights)

            covariance_matrices.append(weighted_covariance)
            means.append(weighted_mean)

    return means, covariance_matrices



def plot_surface_with_distance(distance_grid):
    # 归一化距离
    distance_min = distance_grid.min()
    distance_max = distance_grid.max()

    # 设置颜色映射范围为 [min/10, max/10]
    vmin = distance_min / 10
    vmax = distance_max / 10

    # 使用归一化来将distance映射到[0, 1]范围内
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 生成渐变颜色
    normalized_distances = (distance_grid - distance_grid.min()) / (distance_grid.max() - distance_grid.min())
    colors = plt.cm.RdYlGn(1 - normalized_distances)
    # 绘制三次曲面
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制平滑的三次曲面并应用颜色映射
    ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.5, linewidth=0, antialiased=True)

    # 设置颜色色带
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlGn_r), ax=ax, shrink=1, aspect=30)
    cbar.set_label('Variance')

    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['{:.2f}'.format(tick) for tick in ticks])

    # 绘制随机点
    # ax.scatter(random_points[:, 0], random_points[:, 1], z_random_points, color='black', label="Train Point", s=50)

    ppoints = []

    # 添加垂直线
    has_label = False
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # if (i+16) % 33 == 0 and (j+16) % 33 == 0:
            if (i-3) %11 == 0 and (j-3) % 11 == 0:
                # 获取网格点的预测Z值
                z_pred = Z[i, j]+0.01
                # # 绘制垂直线
                # if not has_label:
                #     ax.plot([X[i, j], X[i, j]], [Y[i, j], Y[i, j]], [-0.15, z_pred], color='red', linestyle="-", label="Vertical Line", linewidth=2)
                #     has_label = True
                # else:
                #     ax.plot([X[i, j], X[i, j]], [Y[i, j], Y[i, j]], [-0.15, z_pred], color='red', linestyle="-", linewidth=2)
                # 存储预测点
                ppoints.append([X[i, j], Y[i, j], z_pred])

    # 将预测点转换为 NumPy 数组
    ppoints = np.array(ppoints)

    # 一次性绘制所有散点
    ax.scatter(ppoints[:, 0], ppoints[:, 1], ppoints[:, 2], color='blue', s=70, label="Predicted Point")
    ax.set_box_aspect([1, 1, 1])
    # 设置标签
    ax.set_xlabel('parameter axis X')
    ax.set_ylabel('parameter axis Y')
    ax.set_zlabel('value axis Z')
    # 隐藏坐标轴
    ax.axis('off')
    plt.show()

def plot_all_ellipsoid():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
   
    ppoints = []

    # 添加垂直线
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (i-3) % 11 == 0 and (j-3) % 11 == 0:
                # 获取网格点的预测Z值
                z_pred = Z[i, j]
                # 存储预测点
                ppoints.append([X[i, j], Y[i, j], z_pred])

    # 将预测点转换为 NumPy 数组
    ppoints = np.array(ppoints)

    means, covariance_matrices = fit_ellipsoids(ppoints)

    # 绘制拟合得到的椭球体
    for i, (mean, cov) in enumerate(zip(means, covariance_matrices)):
        has_quiver = i==0
        plot_ellipsoid(mean, cov, ax, alpha=0.2, color='green', has_quiver=has_quiver)

    ax.scatter(ppoints[:, 0], ppoints[:, 1], ppoints[:, 2], color='blue', s=70, label="Predicted Point")
    ax.set_box_aspect([1, 1, 1])
    # 设置标签
    ax.set_xlabel('parameter axis X')
    ax.set_ylabel('parameter axis Y')
    ax.set_zlabel('value axis Z')
    ax.legend()
    # ax.set_title('Smooth Cubic Surface with Random Points and Vertical Lines')
    ax.grid(False)
    # # 隐藏坐标轴
    # ax.axis('off')
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    # 在[-0.2, 0.2]范围内生成20个随机点
    random_points = np.random.rand(12, 2) * 0.4 - 0.2

    # 根据XY坐标排序，确保从左下到右上逐渐升高
    sorted_indices = np.lexsort((random_points[:, 1], random_points[:, 0]))
    random_points = random_points[sorted_indices]

    # 生成Z坐标，确保从左下到右上逐渐升高
    z_random_points = np.linspace(-0.05, 0.05, len(random_points))

    # 高斯过程拟合
    kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (5, 5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(random_points, z_random_points)

    # 生成网格数据
    x = np.linspace(-0.20, 0.20, 100)
    y = np.linspace(-0.20, 0.20, 100)
    X, Y = np.meshgrid(x, y)
    Z = gp.predict(np.vstack((X.ravel(), Y.ravel())).T).reshape(X.shape)

    # 距离计算
    distance_grid = np.zeros_like(Z)
    for point in random_points:
        z_point = np.sin(np.sqrt(point[0]**2 + point[1]**2))  # 假设gp.predict得到的z值
        distance_grid += np.sqrt((X - point[0])**2 + (Y - point[1])**2 + (Z - z_point)**2)

    plot_surface_with_distance(distance_grid)
    # plot_all_ellipsoid()