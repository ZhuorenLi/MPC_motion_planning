import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from Obs_prediction import obs_prediction  # 导入 obs_prediction 函数

# 定义多个障碍物的初始状态：每个障碍物的状态为 [x, y, theta, v, l, w]
obs_list = [
    np.array([[50, 3.5, 0, 15, 4.8, 1.8]]),  # 障碍物 1
    np.array([[60, 10, np.pi / 4, 10, 3.6, 1.5]]),  # 障碍物 2
    np.array([[70, -5, -np.pi / 6, 12, 5.0, 2.0]])  # 障碍物 3
]

dt = 0.1  # 时间步长 (秒)
N_p = 50  # 预测步数

# 调用 obs_prediction 函数获取所有障碍物的预测轨迹
trajectories = obs_prediction(obs_list, dt, N_p)

# 绘制轨迹和椭圆框
plt.figure(figsize=(10, 8))

for i, trajectory in enumerate(trajectories):
    # 提取 x 和 y 坐标用于绘图
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]

    # 绘制轨迹
    plt.plot(x_coords, y_coords, label=f"Obstacle {i+1} Trajectory")
    plt.scatter(x_coords[0], y_coords[0], color="green")  # 起点
    plt.scatter(x_coords[-1], y_coords[-1], color="red")  # 终点

    # 每隔 1 秒绘制一个椭圆框
    ellipse_a = obs_list[i][0, 4] / 2  # 水平半轴长度
    ellipse_b = obs_list[i][0, 5] / 2  # 垂直半轴长度
    steps_per_second = int(1 / dt)  # 每秒包含的时间步数
    for j in range(0, N_p, steps_per_second):
        x_center = trajectory[j, 0]  # 椭圆中心 x 坐标
        y_center = trajectory[j, 1]  # 椭圆中心 y 坐标
        ellipse = Ellipse(
            (x_center, y_center), width=2 * ellipse_a, height=2 * ellipse_b,
            angle=np.degrees(trajectory[j, 2]), color='orange', fill=False, linewidth=2
        )
        plt.gca().add_patch(ellipse)  # 将椭圆添加到图中

# 设置图例和网格
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Multiple Obstacles Trajectories with Elliptical Bounding Boxes")
plt.legend()
plt.grid()
plt.axis('equal')  # 确保 x 和 y 轴比例相同
plt.show()