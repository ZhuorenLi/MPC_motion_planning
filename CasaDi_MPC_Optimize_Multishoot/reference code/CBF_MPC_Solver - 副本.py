import casadi as ca
import math
import numpy as np
import yaml
from  helpers import load_config
PARAMS_FILE = "mpc_parameters.yaml"

class CBFMPCSolver:
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
            print("N_p: ",self.N_p)
        
        # 从配置文件加载车辆参数
        vehicle_params = self.config['vehicle_params']
        dynamics_constraints = self.config['dynamics_constraints']
        tire_params = self.config['tire_params']
        # 车辆基本参数
        self.Veh_l = vehicle_params['Veh_l']
        self.Veh_L = vehicle_params['Veh_L']
        self.Veh_w = vehicle_params['Veh_w']
        self.Cf_0 = vehicle_params['Cf_0']
        self.Cr_0 = vehicle_params['Cr_0']
        self.Veh_m = vehicle_params['Veh_m']
        self.Veh_lf = vehicle_params['Veh_lf']
        self.Veh_lr = vehicle_params['Veh_lr']
        self.Veh_Iz = vehicle_params['Veh_Iz']
        # 轮胎参数
        self.aopt_f = tire_params['aopt_f']
        self.aopt_r = tire_params['aopt_r']
        self.Fymax_f = self.Cf_0 * self.aopt_f / 2
        self.Fymax_r = self.Cr_0 *self. aopt_r / 2
        # 动力学约束
        self.ax_max = dynamics_constraints['ax_max']
        self.ax_min = dynamics_constraints['ax_min']
        self.df_max = dynamics_constraints['df_max']
        self.df_min = dynamics_constraints['df_min']
        self.jerk_min = dynamics_constraints['jerk_min']
        self.jerk_max = dynamics_constraints['jerk_max']
        self.delta_f_dot_min = dynamics_constraints['delta_f_dot_min']
        self.delta_f_dot_max = dynamics_constraints['delta_f_dot_max']
        # 速度约束
        velocity_constraints = self.config['velocity_constraints']
        self.vx_max = velocity_constraints['vx_max']
        self.vx_min = velocity_constraints['vx_min']
        self.vy_max = velocity_constraints['vy_max']
        self.vy_min = velocity_constraints['vy_min']
        self.model_type = self.config['model_type']
        # 测试所有的参数
        print("self.config: ",self.config) 

        # # 定义状态变量
        self.init_states(self.model_type)
        # 定义控制变量
        self.init_controls()

        self.Y_min=ca.SX.sym('Y_min',self.N_p+1)
        self.Y_max=ca.SX.sym('Y_max',self.N_p+1)
        self.X_g = ca.SX.sym('X_g')
        self.Y_g = ca.SX.sym('Y_g')
        self.e_d = ca.SX.sym('e_d')
        self.e_phi = ca.SX.sym('e_phi')
        self.s = ca.SX.sym('s')
        self.r = ca.SX.sym('r')
        self.delta_f = ca.SX.sym('delta_f')
        self.kappa = 0
        epsilon = 1e-6

        # 建立系统模型
        self.init_model(self.model_type)
        # observation: states, reference states, obstacles, lateral constraints
        kinematics_states = 4
        n_output_states = kinematics_states
        n_refstates = kinematics_states
        n_obstacles = 4
        n_lat_constraints = 2
        # self.P = ca.SX.sym('P', n_output_states + n_refstates + n_obstacles*2 + n_lat_constraints, self.N_p+1)
        self.P = ca.SX.sym('P', n_output_states + n_refstates + n_obstacles*2, self.N_p+1)
        # 定义索引
        self.index_ref = n_output_states
        self.index_obstacles = self.index_ref + n_refstates
        # self.index_lat_constraints = self.index_obstacles + n_obstacles*2 
        # 定义状态参数
        self.X[0, :] = self.P[:self.index_ref, 0] # 初始状态 0时刻
        self.X_obstacle = self.P[self.index_obstacles: self.index_obstacles + n_obstacles, :] # 障碍物X坐标,所有预测时刻
        self.Y_obstacle = self.P[self.index_obstacles + n_obstacles:, :] # 障碍物Y坐标,所有预测时刻
        # self.Ymin = self.P[-2,:] #所有预测时刻
        # self.Ymax = self.P[-1,:] #所有预测时刻


        self.obj = 0  # cost function
        self.cbf_slack = ca.SX.sym('cbf_slack', self.N_p)
        self.g = []    # equal constrains
        self.lbg = []  #lower bound of equal constrains
        self.ubg = []  #upper bound of equal constrains
        self.lbx = []  #lower bound of control input
        self.ubx = []  #upper bound of control input
        
        # guess init_control
        # self.init_control = np.array([0.0, 0.0]*self.N_p).reshape(-1, 2).T
        # self.init_control = np.concatenate(self.init_control.reshape(-1, 1),np.zeros((self.N_p,0))) # 初始控制输入和松弛控制输入 
       
        

    def init_states(self,model_type):
        # 定义状态变量
        
        if model_type == 'kinematics':
            # Kinematics
            self.x = ca.SX.sym('x')   # Frenet s
            self.y = ca.SX.sym('y')   # Frenet l
            self.theta = ca.SX.sym('theta')
            self.vx = ca.SX.sym('vx')
            self.states = ca.vertcat(self.x, self.y, self.theta, self.vx)
            self.n_states = self.states.size()[0]
            #test 
            print("self.states.size(): ",self.states.size())  
            print("self.n_states: ",self.n_states)
        elif model_type == 'dynamics':      
            # Dynamics
            self.x = ca.SX.sym('x')
            self.y = ca.SX.sym('y')
            self.theta = ca.SX.sym('theta')
            self.vx = ca.SX.sym('vx')
            self.vy = ca.SX.sym('vy')
            self.v=ca.SX.sym('v')
            self.r = ca.SX.sym('r')
            self.states = ca.vertcat(self.x, self.y, self.theta, self.vx, self.vy, self.r)
            self.n_states = self.states.size()[0]
            #test 
            print("self.states.size(): ",self.states.size())
            print("self.n_states: ",self.n_states)
            # 补充参数
            self.alpha_f = self.delta_f - (self.vy + self.Veh_lf * self.r) / self.vx
            self.alpha_r = - (self.vy - self.Veh_lr * self.r) / self.vx
            self.Cf = self.Fymax_f * 2 * self.aopt_f / (self.aopt_f ** 2 + self.alpha_f ** 2)
            self.Cr = self.Fymax_r * 2 * self.aopt_r / (self.aopt_r ** 2 + self.alpha_r ** 2)
            self.beta = ca.SX.sym('beta')
            self.Fcf = -self.Cf * self.alpha_f
            self.Fcr = -self.Cr * self.alpha_r
    
    def init_controls(self):
        # 定义控制变量
        self.ax = ca.SX.sym('ax')
        self.df = ca.SX.sym('df')
        self.controls = ca.vertcat(self.ax,self.df)
        self.n_controls = self.controls.size()[0]
        #test 
        print("self.controls.size(): ",self.controls.size())
    
    def init_model(self,model_type):
        # 建立系统模型
        if model_type == 'kinematics':
            # Kinematics
            self.rhs = ca.horzcat(self.vx * ca.cos(self.theta),
                                  self.vx * ca.sin(self.theta),
                                  self.vx * ca.tan(self.df) / self.Veh_l,
                                  self.ax)
        elif model_type == 'dynamics':
            # Dynamics
            self.rhs = ca.horzcat((self.vx * ca.cos(self.theta) - self.vy * ca.sin(self.theta)) / (1 - self.kappa * self.y),
                                  self.vx * ca.sin(self.theta) + self.vy * ca.cos(self.theta),
                                  self.r - self.kappa * (self.vx * ca.cos(self.theta) - self.vy * ca.sin(self.theta)) / (1 - self.kappa * self.y),
                                  self.ax,
                                  (self.Fcf + self.Fcr) / self.Veh_m - self.vx * self.r,
                                  (self.Veh_lf * self.Fcf - self.Veh_lr * self.Fcr) / self.Veh_Iz)
        self.f = ca.Function('f', [self.states, self.controls], [self.rhs], ['input_state', 'control_input'], ['rhs'])
        #test 
        print("self.f: ",self.f)
        self.U = ca.SX.sym('U', self.N_p, self.n_controls)
        #test 
        print("self.U: ",self.U)
        self.X = ca.SX.sym('X', self.N_p + 1, self.n_states)
        #test 
        print("self.X: ",self.X)

    def optimize_solver(self):
        # 进行状态预测 单次射击法需要，多重射击法直接添加动力学约束
        # for i in range(self.N_p):
        #     self.f_value = self.f(self.X[i, :], self.U[i, :])
        #     self.X[i + 1, :] = self.X[i, :] + self.f_value * (self.t_vector[i+1]-self.t_vector[i])
        # 首先添加系统动力学约束 多重射击法需要
        for i in range(self.N_p):
            f_value = self.f(self.X[i, :], self.U[i, :])
            next_state = self.X[i, :] + f_value * (self.t_vector[i+1]-self.t_vector[i])
            # 添加约束: X[i+1] = f(X[i], U[i])
            # 对每个状态分量分别添加约束
            for j in range(self.n_states):
                self.g.append(self.X[i+1, j] - next_state[j])
                self.lbg.append(0)
                self.ubg.append(0)

        # cbf constraint 0 ~ N_p -1 （共N_p个）
        safety_margin = 1.5
        degree = 2
        gamma = 0.5
        l_agent = self.Veh_L / 2    
        w_agent = self.Veh_w / 2   
        l_obs = l_agent
        w_obs = w_agent
        # for i in range(self.N_p - 2):
        #     for j in range(self.X_obstacle.shape[0]):  # n_obstacles
        #         is_obstacle = ca.logic_not(ca.is_equal(self.X_obstacle[j, i], 0))
        #         if is_obstacle:
        #             self.g.append(0)
        #             self.lbg.append(0)
        #             self.ubg.append(ca.inf)
        #             continue
        #         else:
        #             print("has obstacle")
        #             relative_s = self.X_obstacle[j, i + 1] - self.X[i + 1, 0] 
        #             relative_l  = self.Y_obstacle[j, i + 1] - self.X[i + 1, 1]               
        #             relative_s_next = self.X_obstacle[j, i + 2] - self.X[i + 2, 0]
        #             relative_l_next = self.Y_obstacle[j, i + 2] - self.X[i + 2, 1]
                    
        #             # cbf_h = ((relative_s ** 2 / ((l_agent + l_obs) ** 2)) +
        #             #          (relative_l ** 2 / ((w_agent + w_obs) ** 2)) - 1 - safety_margin - cbf_slack[i])
        #             # cbf_h_next = ((relative_s_next ** 2 / ((l_agent + l_obs) ** 2)) +
        #             #               (relative_l_next ** 2 / ((w_agent + w_obs) ** 2)) - 1 - safety_margin - cbf_slack[i+1])
        #             # self.g.append(cbf_h_next - cbf_h + gamma * cbf_h)
        #             cbf_h = ((relative_s ** 2 / ((l_agent + l_obs) ** 2)) +
        #                     (relative_l ** 2 / ((w_agent + w_obs) ** 2)) - 1 - safety_margin)
        #             cbf_h_next = ((relative_s_next ** 2 / ((l_agent + l_obs) ** 2)) +
        #                         (relative_l_next ** 2 / ((w_agent + w_obs) ** 2)) - 1 - safety_margin)
        #             self.g.append(cbf_h_next - cbf_h + gamma * cbf_h + cbf_slack[i])
        #             # self.g.append(cbf_h_next - cbf_h + gamma * cbf_h)
        #             self.lbg.append(0)
        #             self.ubg.append(ca.inf)
        # tmp_g =self.g[-1]
        # tmp_lbg = self.lbg[-1]
        # tmp_ubg = self.ubg[-1]
        # self.g.append(tmp_g)
        # self.lbg.append(tmp_lbg)
        # self.ubg.append(tmp_ubg)

        # 添加状态变量的边界
        # 对于每个状态变量和每个时间步
        for i in range(self.N_p + 1):
            # 跳过初始状态（i=0），因为它已经固定为当前状态
            if i > 0:
                # x坐标边界 - 可以根据场景调整
                self.lbx.append(-ca.inf)
                self.ubx.append(ca.inf)
                
                # y坐标边界 - 可以根据场景调整
                self.lbx.append(-2)
                self.ubx.append(6)
                
                # theta边界 - 角度通常在[-π, π]范围内
                self.lbx.append(-0.5*ca.pi)
                self.ubx.append(0.5*ca.pi)
                
                # vx边界 - 速度约束
                self.lbx.append(self.vx_min)
                self.ubx.append(self.vx_max)
        # 添加控制输入约束
        for _ in range(self.N_p):
            self.lbx.append(self.ax_min)
            self.ubx.append(self.ax_max)
        for _ in range(self.N_p):
            self.lbx.append(self.df_min)
            self.ubx.append(self.df_max)
        # # 松弛变量边界
        # for _ in range(self.N_p):
        #     self.lbx.append(0)
        #     self.ubx.append(ca.inf)
        
        # 权重矩阵
        self.Q = np.diag([10, 10, 10, 1])  # 降低速度权重
        Ru11 = 0.1 / (2 ** 2)              # 降低控制权重
        Ru22 = 0.1 / ((30 / 180 * 3.14) ** 2)
        self.Ru = np.diag([Ru11, Ru22])
        Rdu11 = 1.0                        # 降低控制变化权重
        Rdu21 = 0.1
        self.Rdu = np.diag([Rdu11, Rdu21])
        # sum of objective function
        # observation: states, reference states, obstacles, lateral constraints
        # size: n_output_states + n_refstates + n_refweights + n_obstacles*2 + n_lat_constraints
        self.obj = 0
        for i in range(self.N_p):
            # # 状态代价
            # obj_state = ca.mtimes([state_error.T, self.Q, state_error])
            obj_x1 = (self.X[i+1, 0] - self.P[self.index_ref,i+1])**2 * self.Q[0,0]  # 纵向位置权重和参考值
            obj_x2 = (self.X[i+1, 1] - self.P[self.index_ref+1,i+1])**2 * self.Q[1,1]  # 横向偏差权重和参考值
            obj_x3 = (self.X[i+1, 2] - self.P[self.index_ref+2,i+1])**2 * self.Q[2,2]  # 航向角偏差权重和参考值
            obj_x4 = (self.X[i+1, 3] - self.P[self.index_ref+3,i+1])**2 * self.Q[3,3]  # 纵向速度权重和参考值
            # 控制输入项
            obj_u = self.U[i, 0]**2 * self.Ru[0,0] + self.U[i, 1]**2 * self.Ru[1,1]
            if i > 0:    # 控制输入变化obj
                obj_du = ((self.U[i, 0] - self.U[i-1, 0])/self.T_S)**2 * self.Rdu[0,0] + \
                    ((self.U[i, 1] - self.U[i-1, 1])/self.T_S)**2 * self.Rdu[1,1]
                sum_obj = obj_x1 + obj_x2 + obj_x3 + obj_x4 + obj_u + obj_du #+ self.cbf_slack[i]
            else:
                sum_obj = obj_x1 + obj_x2 + obj_x3 + obj_x4 + obj_u #+ self.cbf_slack[i]
            self.obj += sum_obj
        # 修改优化变量：现在包括控制输入、状态和松弛变量
        # 控制输入：reshape(U, -1, 1)
        # 状态变量（排除初始状态）：reshape(X[1:, :], -1, 1)
        # 松弛变量：cbf_slack
        # 注意：初始状态X[0, :]不作为优化变量，因为它是固定的当前状态
        opt_variables = ca.vertcat(
            ca.reshape(self.U, -1, 1),                 # 控制输入
            ca.reshape(self.X[1:, :], -1, 1)          # 状态变量（除了初始状态）   
        )
         # opt_variables self.cbf_slack                             # CBF松弛变量
        nlp_prob = {'f': self.obj, 'x': opt_variables, 'p': self.P, 'g': ca.vertcat(*self.g)}
        opts_setting = {'ipopt.max_iter': 50, 'ipopt.print_level': 2, 'print_time': 1, 'ipopt.acceptable_tol': 1e-4,
                        'ipopt.acceptable_obj_change_tol': 1e-4}

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        return solver

    def optimize(self, ego_state, observation, external_ed_min=None, external_ed_max=None):
        # 构造参数向量c_p
        c_p = np.zeros((self.P.shape[0], self.N_p + 1))
        
        # 填充当前状态
        c_p[:self.index_ref, 0] = ego_state  # 当前状态
        c_p[self.index_ref:self.index_obstacles, :] = observation['reference']  # 参考轨迹
        c_p[self.index_obstacles:, :] = observation['obstacles']  # 障碍物信息
        
        # 创建求解器
        solve = self.optimize_solver()
        
        # 使用求解器求解
        res = solve(p=c_p, lbg=self.lbg, lbx=self.lbx, ubg=self.ubg, ubx=self.ubx)
        
        # 检查求解状态
        status = res['status']
        print("求解状态:", status)
        
        # 从结果中提取控制量和状态量
        optimal_solution = res['x'].full().flatten()
        
        # 重塑控制输入矩阵 (N_p, n_controls)
        control_inputs = optimal_solution[:self.N_p * self.n_controls].reshape(self.N_p, self.n_controls)
        
        # 重塑状态量矩阵 (N_p, n_states)
        optimal_states = optimal_solution[self.N_p * self.n_controls:self.N_p * self.n_controls + self.N_p * self.n_states].reshape(self.N_p, self.n_states)
        
        # 打印调试信息
        print("\n=== 调试信息 ===")
        print(f"预测时域长度 N_p: {self.N_p}")
        print(f"控制输入维度: {self.n_controls}")
        print(f"状态量维度: {self.n_states}")
        print("\n参考轨迹:")
        print(f"参考速度: {observation['reference'][3, :]}")
        print("\n控制输入矩阵:")
        print(control_inputs)
        print("\n状态量矩阵:")
        print(optimal_states)
        
        # 检查权重矩阵
        print("\n权重矩阵:")
        print("状态权重 Q:", self.Q)
        print("控制权重 Ru:", self.Ru)
        print("控制变化权重 Rdu:", self.Rdu)
        
        # 检查约束条件
        print("\n约束条件:")
        print(f"加速度约束: [{self.ax_min}, {self.ax_max}]")
        print(f"转向角约束: [{self.df_min}, {self.df_max}]")
        print(f"速度约束: [{self.vx_min}, {self.vx_max}]")
        
        return control_inputs, optimal_states




