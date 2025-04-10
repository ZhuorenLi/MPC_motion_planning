import numpy as np

def obs_prediction(obs_list, dt, N_p):
    """
    根据输入的多个障碍物初始状态、时间步长和预测步数，计算所有障碍物的预测轨迹。
    
    参数:
        obs_list (list of numpy.ndarray): 每个元素是一个障碍物的初始状态 [x, y, theta, v, l, w]
        dt (float): 时间步长 (秒)
        N_p (int): 预测步数
    
    返回:
        trajectories (list of numpy.ndarray): 每个障碍物的预测轨迹，每行表示 [x, y, theta, v, l, w]
    """
    trajectories = []
    
    for obs in obs_list:
        # 初始化轨迹存储
        trajectory = [obs]

        # 基于运动学递推预测轨迹
        for _ in range(N_p):
            current_state = trajectory[-1]  # 获取当前状态
            x, y, theta, v, l, w = current_state[0]  # 解包当前状态
            
            # 计算下一时刻的状态
            next_x = x + v * np.cos(theta) * dt
            next_y = y + v * np.sin(theta) * dt
            next_theta = theta  # 方向不变
            next_v = v  # 速度不变
            
            # 更新状态
            next_state = np.array([[next_x, next_y, next_theta, next_v, l, w]])
            trajectory.append(next_state)

        # 将轨迹转换为 NumPy 数组，并添加到结果列表中
        trajectory = np.vstack(trajectory)
        trajectories.append(trajectory)

    return trajectories