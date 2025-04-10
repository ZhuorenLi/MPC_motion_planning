import casadi as ca
import casadi.tools as ca_tools
import math
import numpy as np
import yaml
from  helpers import load_config
PARAMS_FILE = "mpc_parameters.yaml"

class MPC_optimize:
    def __init__(self):
        self.config = load_config(PARAMS_FILE)
        mpc_params = self.config['mpc_params']
        self.T_horizon = mpc_params['horizon']
        self.T_S = mpc_params['T_S']
        self.pre_time = mpc_params['pre_time']
        self.T_L = mpc_params['T_L']
        self.t_ratio = mpc_params['t_ratio']
        self.is_variable_time = mpc_params['is_variable_time']
        if self.is_variable_time==True:
            t1 = np.arange(0, self.T_horizon * self.t_ratio, self.T_S, dtype=float)
            t2 = np.arange(t1[-1] + self.T_L, t1[-1] + self.T_L + self.T_horizon * (1-self.t_ratio), self.T_L)
            N_p1 = len(t1)
            N_p2 = len(t2)
            self.N_p = N_p1 + N_p2
            self.t_vector = np.concatenate((t1, t2))
            # # test
            # print("N_p1: ",N_p1)
            # print("N_p2: ",N_p2)
            # print("N_p: ",self.N_p)
            # print("t_vector: ",self.t_vector)
        else:
            self.t_vector = np.arange(0, self.T_horizon+self.T_S, self.T_S, dtype=float)
            self.N_p = len(self.t_vector)-1
            # # test
            print("t_vector: ",self.t_vector)
            print("N_p: ",self.N_p, "self.T_S: ",self.T_S)
        
        # 从配置文件加载车辆参数
        vehicle_params = self.config['vehicle_params']
        dynamics_constraints = self.config['dynamics_constraints']
        tire_params = self.config['tire_params']
        # 车辆基本参数
        self.Veh_l = vehicle_params['Veh_l']
        self.Veh_L = vehicle_params['Veh_L']
        self.Veh_W = vehicle_params['Veh_W']
        self.Veh_m = vehicle_params['Veh_m']
        self.Veh_lf = vehicle_params['Veh_lf']
        self.Veh_lr = vehicle_params['Veh_lr']
        self.Veh_Iz = vehicle_params['Veh_Iz']
        # 轮胎参数
        self.aopt_f = tire_params['aopt_f']
        self.aopt_r = tire_params['aopt_r']
        self.Cf_0 = tire_params['Cf_0']
        self.Cr_0 = tire_params['Cr_0']
        self.Fymax_f = self.Cf_0 * self.aopt_f / 2
        self.Fymax_r = self.Cr_0 *self. aopt_r / 2
        # 动力学约束
        self.vy_max = dynamics_constraints['vy_max']
        self.vy_min = dynamics_constraints['vy_min']
        self.jerk_min = dynamics_constraints['jerk_min']
        self.jerk_max = dynamics_constraints['jerk_max']
        self.df_dot_min = dynamics_constraints['df_dot_min'] * np.pi/180
        self.df_dot_max = dynamics_constraints['df_dot_max'] * np.pi/180
        # 运动学约束
        kinematics_constraints = self.config['kinematics_constraints']
        self.vx_max = kinematics_constraints['vx_max']
        self.vx_min = kinematics_constraints['vx_min']
        self.ax_max = kinematics_constraints['ax_max']
        self.ax_min = kinematics_constraints['ax_min']
        self.df_max = kinematics_constraints['df_max'] * np.pi/180
        self.df_min = kinematics_constraints['df_min'] * np.pi/180
        # self.df_max = np.pi/18
        # self.df_min = -np.pi/18
        self.Y_max = kinematics_constraints['Y_max']
        self.Y_min = kinematics_constraints['Y_min']
        
        self.model_type = self.config['model_type']
        # 测试所有的参数
        print("self.config: ",self.config) 

        self.num_states = 4
        self.num_controls = 2

    def initialize_constraints(self,obs_trajectories):
        lbg = []
        ubg = []
        lbx = []
        ubx = []

        for _ in range(self.N_p):
            # control variables
            lbx.append(self.df_min)
            ubx.append(self.df_max)
            lbx.append(self.ax_min)
            ubx.append(self.ax_max)

        for j in range(self.N_p+1): # state variables
            lbx.append(-np.inf)        #x
            ubx.append(np.inf)
            lbx.append(self.Y_min)     #y
            ubx.append(self.Y_max)
            lbx.append(-np.inf)        #phi
            ubx.append(np.inf)
            lbx.append(self.vx_min)    #vx
            ubx.append(self.vx_max)

        for i in range(self.N_p+1):
            lbg.append(0.0)             # sys dyn
            lbg.append(0.0)
            lbg.append(0.0)
            lbg.append(0.0)
            ubg.append(0.0)
            ubg.append(0.0)
            ubg.append(0.0)
            ubg.append(0.0)
        for i in range(self.N_p):
            if i== 0:
                continue
            lbg.append(self.df_dot_min * self.T_S) #ddf
    #         lbg.append(-np.inf) #dax
            ubg.append(self.df_dot_max * self.T_S)
    #         ubg.append(np.inf)
        # # obs direct constraints
        # for _ in range(self.N_p+1):
        #     for j in range(obs_trajectories.shape[0]):
        #         lbg.append(0.2)
        #         ubg.append(np.inf)
        # obs cbf constraints
        for _ in range(self.N_p):
            for j in range(len(obs_trajectories)):
                lbg.append(0.0)
                ubg.append(np.inf)

        return lbg, ubg, lbx, ubx

    def optimize_problem(self, ego_state, ref_state, obs_trajectories):
        # 定义状态变量
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('phi')    
        vx = ca.SX.sym('vx')      
        states = ca.vertcat(x, y, phi, vx)
        n_states = states.size()[0]

        # 定义控制变量
        df = ca.SX.sym('df')
        ax = ca.SX.sym('ax')
        controls = ca.vertcat(df, ax)
        n_controls = controls.size()[0]
        self.num_states = n_states
        self.num_controls = n_controls
        
        rhs = ca.vertcat(vx * ca.cos(phi),
                         vx * ca.sin(phi),
                         vx * ca.tan(df) / self.Veh_l, #轴距
                         ax)

        ## function
        self.f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
        U = ca.SX.sym('U', n_controls, self.N_p)
        X = ca.SX.sym('X', n_states, self.N_p+1)
        # P = ca.SX.sym('P', n_states+2)   # 参数输入（当前为states + x0和xs），
        # 引入reference states
        # P = ca.SX.sym('P', n_states+n_states) # ori
        P = ca.SX.sym('P', n_states+n_states) # ori

        ## 权重系数
        Q_x = 1e1
        Q_y = 1e5
        Q_phi = 3e5
        Q_vx = 1e4
        Q = np.diag([Q_x, Q_y, Q_phi, Q_vx]) # 状态权重
        # Q = 1000
        DQ_x = 1
        DQ_y = 1
        DQ_phi = 1
        DQ_vx = 1
        DQ = np.diag([DQ_x, DQ_y, DQ_phi, DQ_vx]) # 状态变化率权重
        R_df = 1e4
        R_ax = 1e4
        R = np.array([[R_df, 0.0], [0.0, R_ax]])# 控制输入权重
        DR_df = 1e5
        DR_ax = 1e2
        DR = np.array([[DR_df, 0.0], [0.0, DR_ax]]) # 控制输入变化率权重)

        ### define
        ### define the states within the horizon
        obj = 0

        g = []
        g.append(X[:, 0]-P[:n_states])

        Ulast = np.array([0, 0])
        aa = 0.0
        for i in range(self.N_p):
            # cost function  
            ref_X = aa* ref_state[i+1, :] + (1-aa)*P[n_states:n_states*2]
            # obj_X = ca.mtimes([(X[:, i]-P[n_states:n_states*2]).T, Q, X[:, i]-P[n_states:n_states*2]])
            obj_X = ca.mtimes([(X[:, i]-ref_X).T, Q, X[:, i]-ref_X])
            obj_U = ca.mtimes([U[:, i].T, R, U[:, i]])
            if i > 0:
                obj_dU = ca.mtimes([(U[:, i]-U[:, i-1]).T, DR, U[:, i]-U[:, i-1]])
            else:
                obj_dU = ca.mtimes([(U[:, i]-Ulast).T, DR, U[:, i]-Ulast])
            obj = obj + obj_X + obj_U + obj_dU
            # add equal constrains (system dynamics)
            x_next = self.f(X[:, i], U[:, i])*self.T_S + X[:, i]
            g.append(X[:, i+1]-x_next)   


        for i in range(self.N_p):
            if i == 0 :
                continue
                g.append(U[0, i]-Ulast[0]) #ddf
            else:
                g.append(U[0, i]-U[0, i-1]) #ddf
        #         g.append(U[1, i]-U[1, i-1]) #dax
            # add obs_trajectories avoidance constraint

        ego_hl = self.Veh_L/2
        ego_hw = self.Veh_W/2
        safe_X = 7.0
        safe_Y = 3.0
        safe_disl = 1.0
        safe_disw = 0.5
        # for i in range(self.N_p+1):   # direct constraints
        #     for j in range(obs_trajectories.shape[0]):
        #         obs_x = obs_trajectories[j, 0]
        #         obs_y = obs_trajectories[j, 1]
        #         obs_hl = obs_trajectories[j, 2]/2
        #         obs_hw = obs_trajectories[j, 3]/2
        #         safe_X = ego_hl + obs_hl + safe_disl
        #         safe_Y = ego_hw + obs_hw + safe_disw
        #         g.append(ca.sqrt(((X[0, i]-obs_x)**2)/(safe_X**2)+(X[1, i]-obs_y)**2/(safe_Y**2)-1))
        gamma = 1.00
        for i in range(self.N_p):   # cbf constraints
            for j in range(len(obs_trajectories)):
                # 获取第 j 个障碍物在第 i 步的状态 [x, y, theta, v, l, w]
                obs_state = obs_trajectories[j][i]
                obs_x = obs_state[0]
                obs_y = obs_state[1]
                obs_theta = obs_state[2]
                obs_v = obs_state[3]
                obs_l = obs_state[4]
                obs_w = obs_state[5]
                obs_hl = obs_l/2
                obs_hw = obs_w/2
                safe_X = ego_hl + obs_hl + safe_disl
                safe_Y = ego_hw + obs_hw + safe_disw
                h_func = ((X[0, i]-obs_x)**2)/(safe_X**2)+(X[1, i]-obs_y)**2/(safe_Y**2)-1
                h_func_next = ((X[0, i+1]-obs_x)**2)/(safe_X**2)+(X[1, i+1]-obs_y)**2/(safe_Y**2)-1
                h_dot = (h_func_next - h_func) 
                g.append(h_func)
                # g.append(gamma *h_func + h_dot)
                
        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
        nlp_prob = {'f': obj, 'x': opt_variables, 'p':P, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 5   , 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        return solver


    def generate_ref_path(self, x0, xs):
        Tall = self.T_horizon
        # t = np.linspace(0, T, self.N_p+1)
        T = 3.0
        T2 = Tall - T
        dt = 0.1
        N_p1 = int(T/dt)
        N_p2 = int(T2/dt)
        t = np.linspace(0, T, N_p1)
        t2 = np.linspace(T, Tall, N_p2+1)
        xt = xs.copy()
        xt[0,0] = xs[3,0] * T + x0[0,0]
        # 五次多项式系数计算 (以x坐标为例)
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        b_x = np.array([x0[0,0], x0[3,0], 0, xt[0,0], xt[3,0], 0])  # x位置\速度\加速度
        b_y = np.array([x0[1,0], 0, 0, xt[1,0], 0, 0])  # y位置\速度\加速度
        # 解多项式系数
        coeff_x = np.linalg.solve(A, b_x)
        coeff_y = np.linalg.solve(A, b_y)
        # 生成轨迹
        x_traj = coeff_x  [0] + coeff_x  [1]*t + coeff_x  [2]*t**2 + coeff_x  [3]*t**3 + coeff_x  [4]*t**4 + coeff_x  [5]*t**5
        y_traj = coeff_y  [0] + coeff_y  [1]*t + coeff_y  [2]*t**2 + coeff_y  [3]*t**3 + coeff_y  [4]*t**4 + coeff_y  [5]*t**5
        # 计算速度分量
        vx_traj = coeff_x  [1] + 2*coeff_x  [2]*t + 3*coeff_x  [3]*t**2 + 4*coeff_x  [4]*t**3 + 5*coeff_x  [5]*t**4
        vy_traj = coeff_y  [1] + 2*coeff_y  [2]*t + 3*coeff_y  [3]*t**2 + 4*coeff_y  [4]*t**3 + 5*coeff_y  [5]*t**4
        # 计算航向角
        phi_traj = np.arctan2(vy_traj, vx_traj) * 180 / np.pi
        # 计算合速度
        # v_traj = np.sqrt(vx_traj**2 + vy_traj**2)
        v_traj = np.full(N_p1, xs[3,0])
        st_y = y_traj[-1]
        st_v = v_traj[-1]
        st_phi = phi_traj[-1]
        # 计算t2时刻的x值（线性延伸）
        x_t2 = x_traj[-1] + st_v * (t2 - t[-1])
        x_traj = np.concatenate((x_traj, x_t2))
        y_traj = np.concatenate((y_traj, np.full_like(t2, st_y)))
        phi_traj = np.concatenate((phi_traj, np.full_like(t2, st_phi)))
        v_traj = np.concatenate((v_traj, np.full_like(t2, st_v)))
        # 组合所有轨迹信息
        trajectory = np.column_stack((x_traj, y_traj, phi_traj, v_traj))

        # 返回完整的轨迹数组
        return trajectory