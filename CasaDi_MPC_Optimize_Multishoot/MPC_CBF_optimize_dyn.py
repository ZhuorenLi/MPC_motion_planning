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
        self.Veh_w = vehicle_params['Veh_w']
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
        self.kappa = 0  
        # self.df_max = np.pi/18
        # self.df_min = -np.pi/18
        self.Y_max = kinematics_constraints['Y_max']
        self.Y_min = kinematics_constraints['Y_min']
        
        self.model_type = self.config['model_type']
        # 测试所有的参数
        print("self.config: ",self.config) 

        self.num_states = 6
        self.num_controls = 2

    def initialize_constraints(self):
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
            lbx.append(self.vy_min)    #vy
            ubx.append(self.vy_max)
            lbx.append(-np.inf)    #r
            ubx.append(np.inf)

        for i in range(self.N_p+1):
            lbg.append(0.0)             # sys dyn
            lbg.append(0.0)
            lbg.append(0.0)
            lbg.append(0.0)
            lbg.append(0.0)
            lbg.append(0.0)
            ubg.append(0.0)
            ubg.append(0.0)
            ubg.append(0.0)
            ubg.append(0.0)
            ubg.append(0.0)
            ubg.append(0.0)
            if i > 0 and i < self.N_p:
                lbg.append(self.df_dot_min*self.T_S ) #ddf
                lbg.append(self.jerk_min*self.T_S) #dax
                ubg.append(self.df_dot_max*self.T_S)
                ubg.append(self.jerk_max*self.T_S)
            # obs
        for _ in range(self.N_p+1):
            lbg.append(1)
            ubg.append(np.inf)

        return lbg, ubg, lbx, ubx

    def optimize_problem(self, ego_state, ref_state,obstacle):
        # 定义状态变量
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('phi')    
        vx = ca.SX.sym('vx')      
        vy = ca.SX.sym('vy')
        r = ca.SX.sym('r')
        states = ca.vertcat(x, y, phi, vx, vy, r)
        n_states = states.size()[0]

        # 定义控制变量
        df = ca.SX.sym('df')
        ax = ca.SX.sym('ax')
        controls = ca.vertcat(df, ax)
        n_controls = controls.size()[0]
        self.num_states = n_states
        self.num_controls = n_controls
        
        alpha_f = df - (vy + self.Veh_lf * r) / vx
        alpha_r = - (vy - self.Veh_lr * r) / vx
        Cf = self.Fymax_f * 2 * self.aopt_f / (self.aopt_f ** 2 + alpha_f ** 2)
        Cr = self.Fymax_r * 2 * self.aopt_r / (self.aopt_r ** 2 + alpha_r ** 2)

        Fcf = -Cf * alpha_f
        Fcr = -Cr * alpha_r


        rhs = ca.vertcat(vx * ca.cos(phi) - vy * ca.sin(phi),                                             #x
                         vx * ca.sin(phi) + vy * ca.cos(phi),                                             #y
                         r ,                                                                              #phi
                         ax + r * vy,                                                                     #vx
                         -r*vx + 2/self.Veh_m*(Fcf * ca.cos(df)+Fcr),                                     #vy
                         2/self.Veh_Iz*(self.Veh_lf*Fcf - self.Veh_lr*Fcr))                                #r
        
        # rhs = ca.vertcat(vx * ca.cos(phi) - vy * ca.sin(phi),                                             #x
        #                  vx * ca.sin(phi) + vy * ca.cos(phi),                                             #y
        #                  r - self.kappa * (vx * ca.cos(phi) - vy * ca.sin(phi)) / (1 - self.kappa * y),   #phi
        #                  ax + r * vy,                                                                     #vx
        #                  -r*vx +(Fcf+Fcr)/self.Veh_m,                                                     #vy
        #                  (self.Veh_lf*Fcf - self.Veh_lr*Fcr)/self.Veh_Iz)                                 #r

        ## function
        self.f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
        U = ca.SX.sym('U', n_controls, self.N_p)
        X = ca.SX.sym('X', n_states, self.N_p+1)
        # P = ca.SX.sym('P', n_states+2)   # 参数输入（当前为states + x0和xs），
        # 引入reference states
        # P = ca.SX.sym('P', n_states+n_states) # ori
        P = ca.SX.sym('P', n_states+n_states) # ori

        ## 权重系数
        Q_x = 10
        Q_y = 1e5
        Q_phi = 1e3
        Q_vx = 1e3
        Q_vy = 1
        Q_r = 1
        Q = np.diag([Q_x, Q_y, Q_phi, Q_vx, Q_vy, Q_r]) # 状态权重
        # Q = 1000
        DQ_x = 10
        DQ_y = 100
        DQ_phi = 1e3
        DQ_vx = 100
        DQ_vy = 1
        DQ_r = 1
        DQ = np.diag([DQ_x, DQ_y, DQ_phi, DQ_vx, DQ_vy, DQ_r]) # 状态变化率权重
        R_df = 1e3
        R_ax = 1e3
        R = np.array([[R_df, 0.0], [0.0, R_ax]])# 控制输入权重
        DR_df = 5e3
        DR_ax = 5e2
        DR = np.array([[DR_df, 0.0], [0.0, DR_ax]]) # 控制输入变化率权重)

        ### define
        ### define the states within the horizon
        obj = 0
        g = []
        g.append(X[:, 0]-P[:n_states])
        safe_dis = 0.2
        for i in range(self.N_p):
            # cost function     
            obj_X = ca.mtimes([(X[:, i]-P[n_states:n_states*2]).T, Q, X[:, i]-P[n_states:n_states*2]])
            obj_U = ca.mtimes([U[:, i].T, R, U[:, i]])
            if i > 0:
                obj_dU = ca.mtimes([(U[:, i]-U[:, i-1]).T, DR, U[:, i]-U[:, i-1]])
            else:
                obj_dU = 0
            obj = obj + obj_X + obj_U + obj_dU
            # add equal constrains (system dynamics)
            x_next = self.f(X[:, i], U[:, i])*self.T_S + X[:, i]
            g.append(X[:, i+1]-x_next)      
            if i > 0:
                g.append(U[0, i]-U[0, i-1])  #ddf
                g.append(U[1, i]-U[1, i-1])  #dax
            # add obstacle avoidance constraint
        # for i in range(self.N_p+1):
        #     dis_obs = (X[0, i]-P[n_states*2])**2 + (X[1, i]-P[n_states*2+1])**2
        #     g.append(dis_obs)
        obs_x = 200
        obs_y = 3.5
        obs_x = obstacle[0]
        obs_y = obstacle[1]
        safe_X = 4.0
        safe_Y = 1.0
        for i in range(self.N_p+1):
            g.append(ca.sqrt(((X[0, i]-obs_x)**2)/(safe_X**2)+(X[1, i]-obs_y)**2/(safe_Y**2)-1))

        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
        nlp_prob = {'f': obj, 'x': opt_variables, 'p':P, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 5   , 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        return solver

    # def find_nearest_ref_point(self, x0, last_index):
    #     min_distance = float('inf')
    #     nearest_point = None
    #     k = 7 #预瞄距离权重
    #     #计算车辆坐标系下预瞄距离
    #     target_speed = 6
    #     smooth_alpha = 0.1
    #     preview_speed = (1-smooth_alpha)*rear_axle_center[3][0] + smooth_alpha*target_speed
    #     preview_range_new = (k * np.abs(preview_speed)) / (1 + np.abs(rear_axle_center[2][0]))
    #     preview_range_old = k * np.abs(rear_axle_center[3][0])
        
    #     # print("preview_new = ", preview_range_new)
    #     # print("preview_old = ", preview_range_old)

    #     for idx, point in enumerate(self.resampled_path_points[:]):

    #         #计算两点之间的距离，求最小值
    #         distance = np.sqrt((rear_axle_center[0][0]-point.x)**2 + (rear_axle_center[1][0]-point.y)**2)
            
    #         if distance < min_distance:
    #             min_distance = distance
    #             nearest_point = point
    #             result = idx
    #         # else:
            
    #     preview_idx = int(preview_range_new) #0.25 为全局路采样距离
    #     result = result + preview_idx
    #     return nearest_point, result