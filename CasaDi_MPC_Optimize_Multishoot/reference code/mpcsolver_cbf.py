import casadi as ca
import math
import numpy as np
from  helpers import load_config
from numpy import sin, cos, tan, arctan2, sqrt
Vehicle_PARAMS_FILE = "mpc_mie_parameters.yaml"
from Utils_dynamic_json import *
from casadi import logic_and
class MPCSolver:
    def __init__(self,timeconfig):  # Add other necessary parameters


        self.horizon = 1
        self.T = 0.1
        self.N = 20
        self.pre_time=0.1
        #self.vehicle_params = vehicle_params
        self.vx_max = 40
        self.vx_min = -40
        self.vy_max = 40
        self.vy_min = -40
        # self.Y_max = 4.8
        # self.Y_min = -0.8
        if timeconfig==1:
            Ts2 = 0.1
        else:
            Ts2=0.1
        T = 0.1
        t1 = np.arange(0, T * 11, T, dtype=float)
        t2 = np.arange(t1[-1] + Ts2, Ts2 * 11 + t1[-1], Ts2)
        self.t_vector = np.concatenate((t1, t2))
        self.ax_max = 2 * math.sqrt(2)
        self.ax_min = -2 * math.sqrt(2)
        self.df_max = np.pi / 6
        self.df_min = -np.pi / 6
        self.Veh_l = 2.65
        self.lf = 1.19
        self.lr = self.Veh_l - self.lf
        self.m = 1520
        self.Iz = 2640
        self.Cf_0 = -155495
        self.Cr_0 = -155495
        self.vehicle_main_params = load_config(Vehicle_PARAMS_FILE)
        self.Veh_m = self.vehicle_main_params['Veh_m']
        self.Veh_lf = self.vehicle_main_params['Veh_lf']
        self.Veh_lr = self.vehicle_main_params['Veh_lr']
        self.Veh_Iz = self.vehicle_main_params['Veh_Iz']
        self.jerk_min = self.vehicle_main_params['jerk_min']
        self.jerk_max = self.vehicle_main_params['jerk_max']
        self.delta_f_dot_min = self.vehicle_main_params['delta_f_dot_min']
        self.delta_f_dot_max = self.vehicle_main_params['delta_f_dot_max']
        self.vx = ca.SX.sym('vx')
        self.vy = ca.SX.sym('vy')
        self.v=ca.SX.sym('v')
        self.pd = ca.SX.sym('pd')
        self.p = ca.SX.sym('p')
        self.Y_min=ca.SX.sym('Y_min',self.N+1)
        self.Y_max=ca.SX.sym('Y_max',self.N+1)
        self.X_g = ca.SX.sym('X_g')
        self.Y_g = ca.SX.sym('Y_g')
        self.e_d = ca.SX.sym('e_d')
        self.e_phi = ca.SX.sym('e_phi')
        self.s = ca.SX.sym('s')
        self.r = ca.SX.sym('r')
        self.delta_f = ca.SX.sym('delta_f')

        self.aopt_f = 20 * np.pi / 180
        self.aopt_r = 11 * np.pi / 180
        self.Fymax_f = self.Cf_0 * self.aopt_f / 2
        self.Fymax_r = self.Cr_0 *self. aopt_r / 2
        # states = ca.vertcat(vx, vy, pd, p, X_g, Y_g)
        self.states = ca.vertcat(self.e_d, self.e_phi, self.s,self.vx, self.vy, self.r)
        #self.states = ca.vertcat(self.e_d, self.e_phi, self.s,self.v)

        self.n_states = self.states.size()[0]

        self.ax = ca.SX.sym('ax')
        self.df = ca.SX.sym('df')
        self.controls = ca.vertcat(self.ax,self.df)
        self.n_controls = self.controls.size()[0]

        self.af = self.delta_f - (self.vy + self.lf * self.r) / self.vx
        self.ar = - (self.vy - self.lr * self.r) / self.vx
        self.Cf = self.Fymax_f * 2 * self.aopt_f / (self.aopt_f ** 2 + self.af ** 2)
        self.Cr = self.Fymax_r * 2 * self.aopt_r / (self.aopt_r ** 2 + self.ar ** 2)
        self.beta = ca.SX.sym('beta')
        self.Fcf = -self.Cf * self.af
        self.Fcr = -self.Cr * self.ar
        self.kappa = 0
        epsilon = 1e-6

        self.rhs = ca.horzcat(self.vx * sin(self.e_phi) + self.vy * cos(self.e_phi))
        self.rhs = ca.horzcat(self.rhs,
                              self.r - self.kappa * (self.vx * cos(self.e_phi) - self.vy * sin(self.e_phi)) / (
                                          1 - self.kappa * self.e_d))
        self.rhs = ca.horzcat(self.rhs,
                              (self.vx * cos(self.e_phi) - self.vy * sin(self.e_phi)) / (1 - self.kappa * self.e_d))
        self.rhs = ca.horzcat(self.rhs, self.ax)

        alpha_front = (self.vy + self.lf * self.r) / self.vx - self.df
        alpha_rear = (self.vy - self.lr * self.r) / self.vx
        Fcf = self.Cf_0 * alpha_front
        Fcr = self.Cr_0 * alpha_rear
        self.rhs = ca.horzcat(self.rhs, (Fcf + Fcr) / self.Veh_m - self.vx * self.r)
        self.rhs = ca.horzcat(self.rhs,
                              (self.lf * Fcf - self.lr * Fcr) / self.Iz)



        ## function
        self.f = ca.Function('f', [self.states, self.controls], [self.rhs], ['input_state', 'control_input'], ['rhs'])

        self.U = ca.SX.sym('U', self.N, self.n_controls)

        self.X = ca.SX.sym('X', self.N + 1, self.n_states)

        self.P = ca.SX.sym('P', self.n_states + 5+5+16+2+2 , self.N+1 )
        # self.ed_min_cons=ca.SX.sym('ed_min_con',1,self.N)
        # self.ed_max_cons = ca.SX.sym('ed_max_con', 1,self.N)
        self.Q_ed1 = 10/(3.75**2)
        self.k=1
        self.Q_ed2 = 100/(3.75**2)
        self.Q_ephi = 10/((20/180*np.pi)**2)
        self.Q_s = 0
        self.Q_vx = 10/(2**2)

        self.Q_deltaf=100


        self.R = np.array([[1 / (2 ** 2), 0.0], [0.0, 1 / ((30 / 180 * 3.14) ** 2)]])#highcase礼貌让道

        self.Rdu = np.diag([self.P[16, 0]/ (5 ** 2), self.P[17, 0]/ ((60 / 180 * 3.14) ** 2)])#highdjixianhuandao
        # if W_MPC:
        #     if
        # Q_ed=
        ### define
        self.X[0, :] = self.P[:6,0]  # 初始状态
        self.X_obstacle = self.P[16:20,:]  # 障碍物X坐标
        self.Y_obstacle = self.P[20:24,:]  # 障碍物Y坐标
        self.L_obstacle = self.P[24:28,:]  # 障碍物长度
        self.W_obstacle = self.P[28:32,:]  # 障碍物宽度
        self.Y_min = self.P[32,:]  # 横向约束下限
        self.Y_max = self.P[33,:]  # 横向约束上限

        for i in range(self.N):

            self.f_value = self.f(self.X[i, :], self.U[i, :])
            self.X[i + 1, :] = self.X[i, :] + self.f_value * (self.t_vector[i+1]-self.t_vector[i])


        self.obj = 0  #### cost
        cbf_slack = ca.SX.sym('cbf_slack', self.N)
        self.g = []  # equal constrains
        self.lbg = []
        self.ubg = []

        #safety_margin = 1.7#礼貌让到policy3
        safety_margin =1.5#
        degree = 2
        alpha = 0.5
        l_agent = 5 / 2
        w_agent = 2 / 2
        for i in range(self.N - 1):

            for j in range(self.X_obstacle.shape[0]):
                # 当前时刻的 diffs 和调整
                diffs = self.X[i + 1, 2] - self.X_obstacle[j, i + 1]
                diffey = self.X[i + 1, 0] - self.Y_obstacle[j, i + 1]
                is_within_bounds = ca.if_else(ca.logic_and(diffs <= 20, diffs >= -20), 1, 0)
                is_within_bounds_sv = ca.if_else(ca.logic_and(diffs <= 15, diffs >= 0), 1, 0)
                diffs_clipped = ca.fmax(ca.fmin(diffs, 20), -20)  # 裁剪 diffs 的值到 [-15, 15]
                exp_diff_ev = 1 - ca.exp(0.0274653072167027 * diffs_clipped - 0.143841036225890)
                exp_diff_sv = 1 - ca.exp(-0.0462098120373297 * diffs_clipped - 0.693147180559945)
                                                                                                                                                                                                = self.X[i + 1, 2] + is_within_bounds * (exp_diff_ev * 5)

                X_obstacle_adjust = self.X_obstacle[j, i + 1] + is_within_bounds_sv * exp_diff_sv * 5
                # X_adjust = self.X[i + 1, 2]
                X_obstacle_adjust = self.X_obstacle[j, i + 1]
                diffs = X_adjust - X_obstacle_adjust
                # 下一时刻的 diffs 和调整
                if i < self.N - 2:  # 确保 i+2 不越界
                    diffs_next = self.X[i + 2, 2] - self.X_obstacle[j, i + 2]
                    diffey_next = self.X[i + 2, 0] - self.Y_obstacle[j, i + 2]
                    is_within_bounds_next = ca.if_else(ca.logic_and(diffs_next <= 20, diffs_next >= -20), 1, 0)
                    is_within_bounds_next_sv = ca.if_else(ca.logic_and(diffs_next <= 15, diffs_next >= 0), 1, 0)
                    diffs_clipped = ca.fmax(ca.fmin(diffs_next, 20), -20)
                    exp_diff_ev_next = 1 - ca.exp(0.0274653072167027 * diffs_clipped - 0.143841036225890)
                    exp_diff_sv_next = 1 - ca.exp(-0.0462098120373297 * diffs_clipped - 0.693147180559945)
                    X_adjust_next = self.X[i + 2, 2] + is_within_bounds_next * (exp_diff_ev_next * 5)

                    X_obstacle_adjust_next = self.X_obstacle[j, i + 2] + is_within_bounds_next_sv * (
                            exp_diff_sv_next * 5)
                    # X_adjust_next = self.X[i + 2, 2]
                    X_obstacle_adjust_next = self.X_obstacle[j, i + 2]
                    diffs_next = X_adjust_next - X_obstacle_adjust_next
                else:
                    # 使用最后一个可用时间步
                    diffs_next = self.X[i + 1, 2] - self.X_obstacle[j, i + 1]
                    diffey_next = self.X[i + 1, 0] - self.Y_obstacle[j, i + 1]
                    is_within_bounds_next = ca.if_else(ca.logic_and(diffs_next <= 20, diffs_next >= -20), 1, 0)
                    is_within_bounds_next_sv = ca.if_else(ca.logic_and(diffs_next <= 15, diffs_next >= 0), 1, 0)
                    diffs_clipped = ca.fmax(ca.fmin(diffs_next, 20), -20)
                    exp_diff_ev_next = 1 - ca.exp(0.0274653072167027 * diffs_clipped - 0.143841036225890)
                    exp_diff_sv_next = 1 - ca.exp(-0.0462098120373297 * diffs_clipped - 0.693147180559945)
                    X_adjust_next = self.X[i + 1, 2] + is_within_bounds_next * (exp_diff_ev_next * 5)

                    X_obstacle_adjust_next = self.X_obstacle[j, i + 1] + is_within_bounds_next_sv * exp_diff_sv_next * 5
                    # X_adjust_next = self.X[i + 1, 2]
                    X_obstacle_adjust_next = self.X_obstacle[j, i + 1]
                    diffs_next = X_adjust_next - X_obstacle_adjust_next
                    # 计算障碍物相关的 CBF 条件
                l_obs = self.L_obstacle[j, i + 1] / 2
                w_obs = self.W_obstacle[j, i + 1] / 2
                h = ((diffs ** degree / ((l_agent + l_obs) ** degree)) +
                     (diffey ** degree / ((w_agent + w_obs) ** degree)) - 1 - safety_margin - cbf_slack[i])
                h_next = ((diffs_next ** degree / ((l_agent + l_obs) ** degree)) +
                          (diffey_next ** degree / ((w_agent + w_obs) ** degree)) - 1 - safety_margin - cbf_slack[
                              i + 1])
                # 添加到约束和目标函数
                self.g.append(h_next - h + alpha * h)
                self.lbg.append(0)
                self.ubg.append(ca.inf)
                # 增加 CBF 松弛变量的权重到目标函数
            self.obj += 50000 * cbf_slack[i]


        for i in range(self.N):
            self.obj = self.obj + self.P[11,i] * (self.X[i + 1, 0] - self.P[6,i]) ** 2  # 横向偏差权重和参考值
            self.obj = self.obj + self.P[12,i] * (self.X[i + 1, 1] - self.P[7,i]) ** 2  # 航向角偏差权重和参考值
            self.obj = self.obj + self.P[13,i] * (self.X[i + 1, 2] - self.P[8,i]) ** 2  # 纵向位置权重和参考值
            self.obj = self.obj + self.P[14,i] * (self.X[i + 1, 3] - self.P[9,i]) ** 2  # 纵向速度权重和参考值

            #elf.obj =self. obj + self.Q_deltaf * (self.X[i + 1, 6] - self.P[11]) ** 2
            self.obj = self.obj + ca.mtimes([self.U[i, :], self.R, self.U[i, :].T])
            if i > 0:
                self.obj = self.obj + ca.mtimes([(self.U[i, :] - self.U[i - 1, :]) / self.T, self.Rdu,
                                                 ((self.U[i, :] - self.U[i - 1, :]) / self.T).T])



        for i in range(self.N + 1):
            # self.g.append(self.X[i, 3])  ##强制为0舒适是有点奇怪
            self.g.append(self.X[i, 0])
            self.g.append(self.X[i, 3])
            if i > 0 and i < self.N:
                self.g.append(self.U[i, 0] - self.U[i - 1, 0])
                self.g.append(self.U[i, 1] - self.U[i - 1, 1])



        for j in range(self.N + 1):

            self.lbg.append(-0.8)
            self.ubg.append(8.8)
            self.lbg.append(self.vx_min)
            self.ubg.append(self.vx_max)
            if j > 0 and j < self.N:
                self.lbg.append(-8 * self.T)
                self.ubg.append(6 * self.T)
                self.lbg.append(-np.pi / 3 * self.T)
                self.ubg.append(np.pi / 3 * self.T)


        self.lbx = []
        self.ubx = []
        for _ in range(self.N):
            self.lbx.append(self.ax_min)
            self.ubx.append(self.ax_max)

        for _ in range(self.N):
            self.lbx.append(self.df_min)
            self.ubx.append(self.df_max)

        for _ in range(self.N):
            self.lbx.append(0)
            self.ubx.append(ca.inf)

        opt_variables = ca.vertcat(ca.reshape(self.U, -1, 1), cbf_slack)

        self.nlp_prob = {'f': self.obj, 'x': opt_variables, 'p': self.P, 'g': ca.vertcat(*self.g)}
        self.opts_setting = {'ipopt.max_iter': 50, 'ipopt.print_level': 2, 'print_time': 1, 'ipopt.acceptable_tol': 1e-4,
                        'ipopt.acceptable_obj_change_tol': 1e-4}

        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts_setting)
        self.t0 = 0.0
        # self.x0 = np.array([0, 0, 0, 25, 0, 0, 0]).reshape(-1, 1)  # initial state
        # self.xs = np.array([0, 0, 25, 0, 0]).reshape(-1, 1)  # final state
        self.x0 = np.array([0, 0, 0, 25,0,0]).reshape(-1, 1)  # initial state
        self.xs = np.array([0,0, 0, 25, 0]).reshape(-1, 1)  # final state
        self.x_ref = []
        # xs = np.array([0, 0, 0]).reshape(-1, 1)
        self.u0 = np.array([0, 0] * self.N).reshape(-1, 2)  # np.ones((N, 2)) # controls
        self.xh = [self.x0]  # contains for the history of the state
        self.uh = []
        self.costvalue=[]
        self.th = [self.t0]  # for the time
        self.sim_time = 5
        self.T_sim = 0.05


    def solve(self, init_control, c_p,external_ed_min=None,external_ed_max=None):

        res = self.solver(x0=init_control, p=c_p, lbg=self.lbg, lbx=self.lbx, ubg=self.ubg, ubx=self.ubx)
        # 检查求解是否成功
        status =self.solver.stats()
        if status["success"]:
            print("求解成功!")
            state_out = 1  # 获取优化结果
        else:
            print("求解失败!")
            state_out = 0
            print(f"状态: {status}")
            if status == 'infeasible_problem':
                print("问题不可行，检查约束条件。")
            elif status == 'max_iter_reached':
                print("达到最大迭代次数，可能需要调整参数或约束。")
            elif status == 'error':
                print("发生错误，检查输入和参数设置。")

        control_inputs = ca.reshape(res['x'], self.N, self.n_controls+1)
        self.costvalue=res['f']

        return control_inputs,state_out

