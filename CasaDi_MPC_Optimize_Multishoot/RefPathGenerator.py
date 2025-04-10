import numpy as np

class RefPathGenerator: 
    def __init__(self):
        self.ref_global = None
        self.step_x = None
        self.ref_len = None

    def define_ref_path(self, x0, xs, dt):

        global_x_len = (xs[0] - x0[0])
        # self.step_x = xs[3] * dt
        self.step_x = 1
        if xs[0] > x0[0]:
            global_x = np.arange(x0[0], xs[0] + self.step_x, self.step_x)  # 加上 delta_x 确保包含终点
        else:
            global_x = np.arange(x0[0], xs[0] - self.step_x, -self.step_x)  # 负步长处理递减情况
        global_y = xs[1] * np.ones_like(global_x)
        global_phi = xs[2] * np.ones_like(global_x)
        global_vx = xs[3] * np.ones_like(global_x)

        self.ref_global = np.array([global_x, global_y, global_phi, global_vx]).T
        self.ref_len = len(self.ref_global)
        return self.ref_global


    def find_ref_traj(self, x0, xs, T_horizon, dt, last_idx):
        """
        # finde nearest point in ref_traj to the x0 --> xref0
        # 从xref0 构建Np+1个点
        """
        N_p = int(T_horizon / dt)
        vs = xs[3]
        smooth_alpha = 0.5
        preview_v = (1-smooth_alpha)*x0[3] + smooth_alpha*vs
        preview_range = preview_v * T_horizon
        preview_idx = int(preview_range / self.step_x)

        seach_l = max(0, last_idx - 5)
        seach_u = min(self.ref_len, last_idx + preview_idx)
        ref_traj_seach = self.ref_global[seach_l:seach_u, :]
        # 计算ref_traj_seach中每个点与x0的距离
        min_dist = np.inf
        for idx, ref_point in enumerate(ref_traj_seach):
            dist = np.sqrt((ref_point[0] - x0[0])**2 + (ref_point[1] - x0[1])**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = seach_l + idx
            else:
                break

        # 按照新的间隔生成 local_ref_traj
        global_indices = np.linspace(min_idx, min_idx + preview_idx, N_p + 1)  # 等间距生成索引
        global_indices = np.clip(global_indices, 0, self.ref_len - 1).astype(int)  # 确保索引在范围内

        # local_ref_traj =  self.ref_global[min_idx:min_idx + N_p + 1, :]
        local_ref_traj = self.ref_global[global_indices, :]

        return local_ref_traj, min_idx












