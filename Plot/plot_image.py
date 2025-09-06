import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot3d(data, file_path, point=None):
    # 创建一个图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')  # 设置图形背景为白色
    ax.set_facecolor('white')  # 设置轴背景为白色

    # 绘制3D曲线
    ax.plot3D(data[0], data[1], data[2], linewidth=1.5, color="Purple")
    if point is not None:
        x,y,z = point
        ax.scatter(x, y, z, color="Red", s=60, label='Catch Point')

    # 添加标题和标签，并设置字体大小为16
    ax.set_title('3D Curve', fontsize=16)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)

    # 设置刻度标签的字体大小
    ax.tick_params(axis='both', which='major', labelsize=12, pad=0)

    # 设置轴面颜色
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # 设置轴格线颜色为浅灰色
    grid_color = (0.8, 0.8, 0.8, 1.0)
    ax.xaxis._axinfo['grid'].update(linestyle="dashed", color=grid_color)
    ax.yaxis._axinfo['grid'].update(linestyle="dashed", color=grid_color)
    ax.zaxis._axinfo['grid'].update(linestyle="dashed", color=grid_color)

    # 保存图像并去掉空白部分
    plt.savefig(file_path, bbox_inches='tight')

    # 显示图形
    plt.show()