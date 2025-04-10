import gymnasium as gym
import matplotlib.pyplot as plt
from fontTools.varLib.models import subList
from jinja2.nodes import Continue


# from pandas.conftest import axis_1

# from examples.dqn_test import observation
# from examples.dqn_test import observation
# from mpc_acados_solver import MPCSolver
from mpcsolver_immkf_highd import MPCSolver, MPCSolver_Tr
from stable_baselines3 import SAC
import numpy as np
from numpy import sin, cos, tan, arctan2, sqrt
from Utils_new_immkfmpc_highd import *
from gymnasium import spaces
from observation_immkfwithoutref import ObservationGenerator
from reward_essay import RewardGenerator
from stable_baselines3.data_display.data_display import Data_Displayer
import pandas as pd
import os
import highway_env
from high_d.src.highd_env_config import get_init_para_for_sb3, HighDEnvOption
from high_d.src.data_management.read_csv import *
import json
from scipy.stats import entropy
from riskcompute import RiskFieldConfig
import math
import pickle
from datetime import datetime
import hashlib
import os
from pathlib import Path
# ... (other imports) ...


class DataRecorder:
    def __init__(self, save_dir="result_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def _get_filename(self, cbfconfig):
        """生成唯一文件名"""
        return self.save_dir /f"policy4case4_{cbfconfig}.pkl"

    def save_data(self, cbfconfig, plot_objects):
        """
        保存绘图数据
        :param policy_config: 策略配置字典，用于唯一标识策略
        :param plot_objects: 包含所有绘图数据的字典
        """
        filename = self._get_filename(cbfconfig)

        # 封装存储数据
        save_data = {
            'policy_config': cbfconfig,
            'metadata': {
                'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_shape': {k: v.shape if hasattr(v, 'shape') else len(v)
                               for k, v in plot_objects.items()}
            },
            'data': plot_objects
        }

        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    def load_data(self, cbfconfig=None):
        """
        加载数据，如不指定policy_config则加载全部
        :return: 按policy_config筛选的数据列表
        """
        all_files = list(self.save_dir.glob("policy_*.pkl"))
        results = []

        for f in all_files:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                if (cbfconfig is None) or (data['policy_config'] == cbfconfig):
                    results.append(data)
        return results


class CustomHighwayEnv(gym.Env):  # Subclass gym.Env for compatibility
    def __init__(self, timeconfig,env_conf,is_record_driving_data: bool, is_plot: bool,policy_config:int,cbfconfig:str):
        # Create the Highway-v0 environment instance
        self.is_plot = is_plot
        self.isppo = False
        self.reward = 0
        self.episode_steps = 0
        self.mpc_solver = MPCSolver(timeconfig)
        seed = 8
        self.T_sim = self.mpc_solver.T_sim
        self.risk_config=RiskFieldConfig()
        self.risk_time=0
        self.danger_time=0
        self.cbfconfig=cbfconfig

        #         break
        self.env = gym.make('highd-v0', render_mode="human", **env_conf)  # 确保环境 ID 正确
        self.intersteps = 0
        self.n_mpcstep = 1
        self.data_recorder=DataRecorder()
        self.plot_data={}
        #self.env = gym.make("highway-v0", render_mode='human')
        #self.env.action_space.seed(seed=seed)
        _,info=self.env.reset(seed=seed)
        self.vx_ref_highd=info["max_v"]
        self.highway_env = self.env.env.env
        self.ttc_cf=self.highway_env.ttc_cf
        self.ttc_cb = self.highway_env.ttc_cb
        self.ttc_tf = self.highway_env.ttc_tf
        self.ttc_tb = self.highway_env.ttc_tb
        self.dis_min_record=[]
        self.risk_record=[]
        self.ttc_record=[]
        self.lat_ttc_record=[]
        self.dis_effect_record=[]
        self.prob=[]
        self.json_file_path = 'reftraj_modena_edgar.json'
        self.ideal_trajectory, self.heading, self.kappa, self.vx, self.ax, self.s, self.Time = generate_trajectory()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.2, high=0.8, shape=(1,), dtype=np.float32)

        self.timeconfig=timeconfig
        self.mpc_solver_T = MPCSolver_Tr()
        self.sac_model = SAC
        self.road = self.env.road
        # ... (other initialization, including SAC model and MPC solver) ...
        self.N = self.mpc_solver.N
        self.T = self.mpc_solver.T
        self.vxmax = 29
        self.u0 = self.mpc_solver.u0
        self.closestpoint_index_old = 0
        #self.data_player = Data_Displayer(isppo=self.isppo, env=self.highway_env)
        self.trajectory_ref = [self.ideal_trajectory.T[0], self.ideal_trajectory.T[1], self.heading, self.kappa,
                               self.vx, self.ax, self.s, self.Time]
        self.cost = []
        self.car_speed_record = []
        self.mu_k_1=[[],[],[]]
        self.mu_k_2=[[],[],[]]
        self.mu_k_3=[[],[],[]]
        self.car_heading_record = []
        self.action_record = []
        self.bemode_record = []
        self.Q_record = []
        self.reward_record = []
        self.env_reward_record = []
        self.deviation_reward = []
        self.deltau_record = []
        self.jerk_record = []
        self.acc_y_record=[]
        self.ay_record=[]
        self.deltaf_dot_record = []
        self.is_record_driving_data = is_record_driving_data
        self.data_record_dir = "./data_record/sac/"
        self.normalization_lims = np.concatenate([[0
                                                      , 0.4],
                                                  [0, 1], ]

                                                 ).transpose()
        self.sigmas = [0.1, 0.5]
        self.policy_config=policy_config
        self.t0 = self.mpc_solver.t0
        self.th = []
        self.delta_f_dot_min = self.mpc_solver.delta_f_dot_min
        self.delta_f_dot_max = self.mpc_solver.delta_f_dot_max
        self.jerk_min = self.mpc_solver.jerk_min
        self.jerk_max = self.mpc_solver.jerk_max
        self.x_ref = self.mpc_solver.x_ref
        self.xh = self.mpc_solver.xh
        self.xs = self.mpc_solver.xs
        self.xs = np.tile(self.xs, (1, 21))
        self.last_ax = 0
        self.last_delta_f = 0

        self.Q = np.array([0.1/(0.5**2), 1.0/((20/180*3.14)**2), 0.0, 10/(2**2), 0.1/((10/180*3.14)**2)]).reshape(-1, 1)
        self.R = 1
        self.Rdu =1
        self.update = False
        self.last_lane=self.highway_env.vehicle.lane_index[2]
        self.lane_change = 0
        self.state_maneger1 = StateManager()
        self.state_maneger2 = StateManager()
        self.state_maneger3 = StateManager()
        self.state_maneger4 = StateManager()
        # self.state_maneger5 = StateManager()
        self.state_maneger_EV = StateManager()
        self.accdot = 0
        self.ddot = 0
        self.jerk = 0
        self.delta_f_dot = 0
        self.new_u = [0.0, 0.0]
        self.ed_refC = self.getReference(4.0)
        self.ed_refT = self.getReference(0.0)
        self.state_manegers = [self.state_maneger1, self.state_maneger2, self.state_maneger3, self.state_maneger4]
        self.x_list = []
        self.y_list = []
        self.vx_list = []
        self.vy_list = []
        self.x_svlist = []
        self.y_svlist = []
        self.vx_svlist = []
        self.vy_svlist = []
        self.x_sv0list = []
        self.y_sv0list = []
        self.vx_sv0list = []
        self.vy_sv0list = []
        self.x_sv1list = []
        self.y_sv1list = []
        self.vx_sv1list = []
        self.vy_sv1list = []
        self.heading_record=[]
        self.filecounter = 4
        self.reset()

    def step(self, actions):

        self.episode_steps += 1

        actions = actions.reshape(1, 1)
        action = actions[0, 0]

        if self.update:
            self.Q[0] = actions[0]/ (0.5 ** 2)
            self.Q[3] = actions[1]/ (2 ** 2)
            self.R=actions[2]*100
            self.Rdu=actions[3]*100

        observation_set = []
        self.states_buffer = []
        for i in range(self.n_mpcstep):
            # print(f'Q_ed:{action}')
            # print(f'i:{i}')
            self.intersteps += 1
            self.steps += 1
            # self.obstacles = self.env.road.vehicles[1:50]
            if len(self.states) == 2:
                self.obstacles = self.states[0][1:5]
                self.ego_v = sqrt(self.states[0][0][5] ** 2 + self.states[0][0][6] ** 2)
                self.ego_ay = self.states[0][0][10]
                self.phi = self.states[0][0][7]
                self.ego_v_x=self.states[0][0][3]
                self.ego_v_y=self.states[0][0][4]
                self.acc_y = self.states[0][0][10]
                self.yawrate=self.states[0][0][8]


            else:
                self.obstacles = self.states[1:5]
                self.ego_v = sqrt(self.states[0][5] ** 2 + self.states[0][6] ** 2)
                self.phi = self.states[0][7]
                self.ego_ay = self.states[0][10]
                self.ego_v_x = self.states[0][3]
                self.ego_v_y = self.states[0][4]
                self.acc_y = self.states[0][10]
                self.yawrate=self.states[0][8]
            # print(self.ego_position)
            self.heading_record.append(self.phi)
            self.ay = (self.mpc_solver.Cf_0 + self.mpc_solver.Cr_0) / (
                        self.mpc_solver.Veh_m * self.ego_v_x) * self.ego_v_y + (
                              (
                                          self.mpc_solver.lf * self.mpc_solver.Cf_0 + self.mpc_solver.lr * self.mpc_solver.Cr_0) / (
                                          self.mpc_solver.Veh_m * self.ego_v_x) - self.ego_v_x) * self.yawrate - self.mpc_solver.Cf_0 / self.mpc_solver.Veh_m * \
                      self.u0[0, 1]
            self.ay_record.append(self.ay)
            non_zero_obstacles=[obs for obs in self.obstacles if np.any(obs[1:3] !=0)]
            num_non_zero_obstacles=len(non_zero_obstacles)
            if num_non_zero_obstacles<2:
                done=True
                observation = np.zeros(35)
                info = {
            'speed': 0.0,
            'crashed': False,
            'on_road': True,  # 假设车辆仍在道路上
            'is_terminal': done,
            'lane_change_count': 0,
            'risk_time': 0.0,
            'danger_time': 0.0,
            'ed_e': [0.0],
            'jerk': 0.0,
            'delta_f_dot': 0.0,
            'acc_y': 0.0,
            'dis_min': float(5),  # 默认设置为无穷大，表示无障碍物
            'dis_effect': float(5),
            'min_TTC': float(5),  # 默认设置为无穷大，表示无障碍物
            'speed_variance': 0.0,
            'lane_changes': 0,
            'closest_vehicle_distance': float(20),  # 默认设置为无穷大
            'closest_vehicle_speed': 0.0,  # 默认速度为0
            'state': [],  # 自定义状态信息，根据实际情况填写
            'lat_ttc': float(5)
        }


                return observation,self.reward,done,True,info
            else:
                pass



            # 初始化横向TTC
            lat_ttc_min = float(5)
            if abs(self.obstacles[0][1])<5.0:
                # 横向位置关系计算
                delta_y = self.obstacles[0][2]
                #same_direction = (np.sign(delta_y) == np.sign(ego.vy)) if ego.vy != 0 else False

                # 仅计算向彼此靠近的情况
                if (delta_y > 0 and self.obstacles[0][4] < 0) or (delta_y < 0 and self.obstacles[0][4] > 0):
                    delta_y_eff = abs(delta_y) - 0.5 * (4)


                    relative_v = abs(self.obstacles[0][4])
                    if relative_v == 0:
                        lat_ttc_min = float(5)

                    ttc = delta_y_eff / relative_v
                    lat_ttc_min = min(lat_ttc_min, ttc)


            self.obstacle_position = self.obstacles[0][1:3] +self.ego_position
            self.obstacle_v_vector=np.array([self.obstacles[0][3]+self.ego_v_x,self.obstacles[0][4]+self.ego_v_y])
            self.ego_v_vector=np.array([self.ego_v_x,self.v_y])
            self.obstacle_phi=self.phi+self.obstacles[0][7]

            risk,dis_effect=self.risk_config.dynamic_risk_potential(self.ego_position,self.ego_v_vector,self.phi,self.obstacle_position,self.obstacle_v_vector,self.obstacle_phi)
            self.dis_effect_record.append(dis_effect)
            if lat_ttc_min<1.5:
                self.risk_time+=self.T_sim
            self.risk_record.append(risk)
            self.ego_position_pre, self.phi_pre = pre_state(self.ego_position, self.phi, self.mpc_solver.pre_time,
                                                          self.x0)
            dis2obs1 = ((self.obstacles[0][1]) ** 2 + (
                         self.obstacles[0][2]) ** 2) ** 0.5

            self.x0 = np.array(self.x0).reshape(-1, 1)
            self.acc_y_record.append(self.acc_y)
            # self.ego_vx = self.x0[3]
            # self.ego_vy = self.x0[4]

            self.distance_min, self.Flagindexenable, self.Flagdistanceenable, self.e_d, self.e_d_dot, self.e_phi, self.e_phi_dot, self.kappa_ref, self.vx_ref, self.ax_ref, self.s_ref, closestpoint_index = updatereflinepointstate(
                self.trajectory_ref, self.x0, self.closestpoint_index_old, self.ego_position_pre, self.phi_pre)
            #self.vx_ref = self.xs[3]
            self.vx_ref = self.vx_ref_highd
            self.ed_ref = self.xs[0]
            self.closestpoint_index_old = closestpoint_index
            # ego_position_next=ego_position+T*x0[0:2]
            self.current_reference_path = get_reference_path(self.ego_position, self.trajectory_ref, self.lookahead)
            # updated_reference_path,dist_array = update_reference_path(current_reference_path, obstacles, safe_distance,ego_position,s_ref,trajectory_ref, x0)
            self.kappa_vector, self.ed_ref, self.ephi_ref, self.s_ref, self.vx_ref, self.StAng_ref, self.ed_max_cons, self.ed_min_cons, self.smax_cons, self.smin_cons, self.Behavior_mode, self.s_obj, self.ed_obj, self.state_manegers, self.accdot, self.ddot, _, self.new_u, self.L_obj, self.W_obj,self.x_pre,self.mu,daptive_para = update_reference_path(
                self.obstacles, self.ego_position, self.s_ref, self.vx_ref, self.trajectory_ref, self.x0, self.phi,
                self.ego_v, self.N, self.T, actions, self.steps, self.state_manegers, self.x0, self.accdot, self.ddot,
                self.new_u, self.ego_ay, self.state_maneger_EV,self.e_d,self.ed_ref,self.timeconfig,dis2obs1,self.policy_config,self.ed_initial,self.ego_v_y)

            self.xs_ori = np.array(
                [self.ed_ref[0], self.ephi_ref[0], self.s_ref[0], self.vx_ref[0], self.StAng_ref[0]]).reshape(-1, 1)


            self.xs = np.array([self.ed_ref, self.ephi_ref, self.s_ref, self.vx_ref, self.StAng_ref])
            self.xs_T = np.array([self.ed_refT, self.ephi_ref, self.s_ref, self.vx_ref, self.StAng_ref])
            self.xs_C = np.array([self.ed_refC, self.ephi_ref, self.s_ref, self.vx_ref, self.StAng_ref])
            self.x0_expended = np.tile(self.x0, (1, self.ed_ref.shape[0]))
            self.Q_expended = np.tile(self.Q, (1, self.ed_ref.shape[0]))
            self.R_expanded=np.tile(self.R,(1,self.ed_ref.shape[0]))
            self.Rdu_expanded=np.tile(self.Rdu,(1,self.ed_ref.shape[0]))
            # print(self.xs)
            self.x_ref.append(self.xs)
            # self.xs=np.array([4,0,30,0,0]).reshape(5,1)
            # Solve MPC with updated weight and current state
            self.c_p = np.concatenate((self.x0_expended, self.xs))
            self.c_p_T = np.concatenate((self.x0_expended, self.xs_T))
            self.c_p_C = np.concatenate((self.x0_expended, self.xs_C))
            # print(self.c_p)
            self.c_p = np.concatenate((self.c_p, self.Q_expended))
            self.c_p_T = np.concatenate((self.c_p_T, self.Q_expended))
            self.c_p_C = np.concatenate((self.c_p_C, self.Q_expended))
            desired_shape = (4, 21)
            s_obj_padded = np.zeros(desired_shape)
            ed_obj_padded = np.zeros(desired_shape)
            L_obj_padded = np.zeros(desired_shape)
            W_obj_padded = np.zeros(desired_shape)
            s_obj_padded[:self.s_obj.shape[0], :self.s_obj.shape[1]] = self.s_obj
            ed_obj_padded[:self.ed_obj.shape[0], :self.ed_obj.shape[1]] = self.ed_obj
            L_obj_padded[:self.L_obj.shape[0], :self.W_obj.shape[1]] = self.L_obj
            W_obj_padded[:self.W_obj.shape[0], :self.W_obj.shape[1]] = self.W_obj
            self.c_p = np.concatenate((self.c_p, s_obj_padded))
            self.c_p = np.concatenate((self.c_p, ed_obj_padded))
            self.c_p = np.concatenate((self.c_p, L_obj_padded))
            self.c_p = np.concatenate((self.c_p, W_obj_padded))

            self.ed_min_cons = np.array(self.ed_min_cons).reshape(-1, 1)
            self.ed_max_cons = np.array(self.ed_max_cons).reshape(-1, 1)
            self.c_p = np.concatenate((self.c_p, self.ed_min_cons.T))
            self.c_p = np.concatenate((self.c_p, self.ed_max_cons.T))
            self.c_p = np.concatenate((self.c_p, self.R_expanded))
            self.c_p = np.concatenate((self.c_p, self.Rdu_expanded))
            self.c_p_T = np.concatenate((self.c_p_T, s_obj_padded))
            self.c_p_T = np.concatenate((self.c_p_T, ed_obj_padded))
            self.c_p_T = np.concatenate((self.c_p_T, L_obj_padded))
            self.c_p_T = np.concatenate((self.c_p_T, W_obj_padded))

            self.c_p_T = np.concatenate((self.c_p_T, self.ed_min_cons.T))
            self.c_p_T = np.concatenate((self.c_p_T, self.ed_max_cons.T))
            self.c_p_C = np.concatenate((self.c_p_C, s_obj_padded))
            self.c_p_C = np.concatenate((self.c_p_C, ed_obj_padded))
            self.c_p_C = np.concatenate((self.c_p_C, L_obj_padded))
            self.c_p_C = np.concatenate((self.c_p_C, W_obj_padded))

            self.c_p_C = np.concatenate((self.c_p_C, self.ed_min_cons.T))
            self.c_p_C = np.concatenate((self.c_p_C, self.ed_max_cons.T))

            self.init_control = ca.reshape(self.u0, -1, 1)
            if self.u0.shape[1] > 2:
                self.init_control = ca.reshape(self.u0, -1, 1)
            # 定义松弛变量，可以根据具体情况初始化
            else:
                slack_variable = np.zeros((20, 1))  # 初始化为0，这只是一个例子
                # 将控制变量和松弛变量合并
                combined_control = np.hstack((self.u0, slack_variable))
                # 将合并后的变量调整为正确的形状
                self.init_control = ca.reshape(combined_control, -1, 1)
            mpc_solution,state_out = self.mpc_solver.solve(self.init_control, self.c_p)

            self.mpc_solver.kappa = self.kappa_vector[0]

            self.cost.append(self.mpc_solver.costvalue)
            cost_value = float(self.mpc_solver.costvalue)
            self.actions = mpc_solution


            self.pre_time = self.mpc_solver.pre_time
            # action = (mpc_solution[0, 0], mpc_solution[0, 1])
            self.th.append(self.t0)

            self.t0, x0, u0, done, self.ego_position, states, r, done, truncated, info = shift_movement(self.env,
                                                                                                        self.T_sim,
                                                                                                        self.t0,
                                                                                                        self.x0,
                                                                                                        self.actions,
                                                                                                        self.trajectory_ref,
                                                                                                        self.closestpoint_index_old,
                                                                                                        self.pre_time,
                                                                                                        self.new_u,
                                                                                                        self.s_obj,
                                                                                                        self.ed_obj,
                                                                                                        self.L_obj,
                                                                                                        self.W_obj,
                                                                                                        self.x_pre,
                                                                                                        self.mu
                                                                                                        )

            if truncated:
                done=True
            x0 = ca.reshape(x0, -1, 1)
            self.xh.append(x0.full())
            self.states = states
            self.x0 = x0
            self.u0 = u0
            self.x_list.append(self.x0[2].full()[0][0])
            self.y_list.append(self.x0[0].full()[0][0])
            self.vx_list.append(float(self.ego_v_x))
            self.vy_list.append(float(self.ego_v_y))
            self.x_svlist.append(float(self.obstacles[0][1]+self.ego_position[0]))
            self.y_svlist.append(float(self.obstacles[0][2]+self.ego_position[1]))
            self.vx_svlist.append(float(self.obstacles[0][3]+self.ego_v_x))
            self.vy_svlist.append(float(self.obstacles[0][4]+self.ego_v_y))
            self.x_sv0list.append(float(self.obstacles[1][1] + self.ego_position[0]))
            self.y_sv0list.append(float(self.obstacles[1][2] + self.ego_position[1]))
            self.vx_sv0list.append(float(self.obstacles[1][3] + self.ego_v_x))
            self.vy_sv0list.append(float(self.obstacles[1][4] + self.ego_v_y))
            self.x_sv1list.append(float(self.obstacles[2][1] + self.ego_position[0]))
            self.y_sv1list.append(float(self.obstacles[2][2] + self.ego_position[1]))
            self.vx_sv1list.append(float(self.obstacles[2][3] + self.ego_v_x))
            self.vy_sv1list.append(float(self.obstacles[2][4] + self.ego_v_y))
            # observation, r, done, truncated, info = self.env.step(action)
            # Extract acceleration and steering angle from MPC solution
            # acceleration, steering_angle = mpc_solution[0]  # Assuming first element contains control inputs

            # Apply control inputs to update vehicle state
            # ... (implement your vehicle dynamics or kinematics update) ...
            # self.state[3] += acceleration  # Example: Update velocity
            # self.state[6] = steering_angle  # Example: Update steering angle

            # Calculate reward, check termination, and update info
            # ... (implement reward function, termination conditions, and info update) ...
            self.observation_gener = ObservationGenerator()

            self.onroad = self.highway_env.vehicle.on_road
            self.crash = self.highway_env.vehicle.crashed
            self.lane_index = self.highway_env.vehicle.lane_index[2]
            if self.lane_index != self.last_lane:
                self.lane_change += 1
                self.last_lane = self.lane_index
            self.ttc_cf = self.highway_env.ttc_cf
            self.ttc_cb = self.highway_env.ttc_cb
            self.ttc_tf = self.highway_env.ttc_tf
            self.ttc_tb = self.highway_env.ttc_tb
            self.ttc_record.append(min(self.ttc_cf,5))
            if self.ttc_cf<2 or lat_ttc_min<1.5:
                self.danger_time+=self.T_sim
            self.lat_ttc_record.append(lat_ttc_min)
            if len(self.state_manegers[0].M) > 0:
                i = 0
                for mu in self.state_manegers[0].MU[-1][0]:
                    if mu != 0:
                        self.mu_k_1[i].append(mu)
                        i = i + 1
            if len(self.state_manegers)>1 and len(self.state_manegers[1].M) > 0:
                i = 0
                for mu in self.state_manegers[1].MU[-1][0]:
                    if mu != 0:
                        self.mu_k_2[i].append(mu)
                        i = i + 1
            N=3
            uniform_dist=[1/N]*N
            mu_k_1=np.array([sublist[-1] if sublist else 0.0 for sublist in self.mu_k_1])
            js_divergence=self.jensen_shannon_divergence(mu_k_1,uniform_dist)
            self.prob.append(daptive_para)
            dis=np.inf
            non_zero_obstacles = self.obstacles[np.any(self.obstacles[:, 1:3] != 0, axis=1)]
            for obs in non_zero_obstacles:
                dis=min(sqrt(obs[1]**2+obs[2]**2),dis)

            self.toend = self.highway_env.config["to_terminal"]
            speed = self.highway_env.vehicle.speed
            reward_env = r
            self.jerk = (self.actions[0, 0] - self.last_ax) / self.T
            self.delta_f_dot = (self.actions[0, 1] - self.last_delta_f) / self.T
            self.last_ax = self.actions[0, 0]
            self.last_delta_f = self.actions[0, 1]
            self.buffer_state = [self.x0[0], self.x0[3] - self.xs_ori[3], self.jerk, self.x0[4]]
            self.states_buffer.append(self.buffer_state)
            # reward=reward_en

            heading = self.highway_env.vehicle.heading
            acc = float(self.actions[0, 0])
            steer = float(self.actions[0, 1])
                                                        cost_value, float(action))

            if self.is_plot:
                self.car_speed_record.append(self.highway_env.vehicle.speed)
                self.car_heading_record.append(self.highway_env.vehicle.heading)
                self.action_record.append(mpc_solution[0, :].full())
                self.dis_min_record.append(dis2obs1)
                self.Q_record.append(actions)
                self.bemode_record.append(float(self.Behavior_mode))
                self.reward_record.append(float(self.reward))
                self.jerk_record.append(self.jerk.full()[0][0])
                self.deltaf_dot_record.append(self.delta_f_dot.full()[0][0])


                if self.is_record_driving_data:
                    self.one_episode_data = pd.DataFrame({'car_speed_record': self.car_speed_record,
                                                          'car_heading_record': self.car_heading_record,
                                                          'action_record': self.action_record,
                                                          'Q_record': self.Q_record,
                                                          'bemode_record': self.bemode_record,
                                                          'reward_record': self.reward_record})
                # self.car_speed_record_sac=self.car_speed_record
            if not self.Flagdistanceenable or not self.Flagindexenable:
                done = True
            if done:
                self.steps = 0
                # 数据记录
                info['lane_change_count'] = self.lane_change
                info['risk_time']=self.risk_time
                info['danger_time']=self.danger_time
                self.save_trajectory("/home/pnc/RLMPC/qyq/qyq/mpc-reinforcement-learning-main/examples/jsonev/testevhighdcase43.json")
                self.save_trajectory_sv("/home/pnc/RLMPC/qyq/qyq/mpc-reinforcement-learning-main/examples/jsonsv/testsvhighdcase43.json")
                # self.save_trajectory_sv0("/home/pnc/RLMPC/qyq/qyq/mpc-reinforcement-learning-main/examples/jsonsv/testsvhighdcase331.json")
                # self.save_trajectory_sv1("/home/pnc/RLMPC/qyq/qyq/mpc-reinforcement-learning-main/examples/jsonsv/testsvhighdcase332.json")

                if self.is_record_driving_data:
                    os.makedirs(self.data_record_dir, exist_ok=True)
                    self.one_episode_data.to_csv(
                        self.data_record_dir + '/' + str(self.intersteps) + '.csv',
                        index=False, sep=',')

                # Create a 2x2 grid of subplots
                if self.is_plot:
                    fig, ax = plt.subplots(8, 2, figsize=(10, 10))
                    Q_record = (np.array(self.Q_record))
                    ed_refplot = [state[0,0] for state in self.x_ref]
                    ed_values = [state[0] for state in self.xh]
                    yawrate=[state[5] for state in self.xh]
                    ed_e = []
                    for i in range(len(ed_refplot)):
                        ed_e.append(np.array(ed_values[i + 1] - ed_refplot[i]))
                    phi_value = [state[1] for state in self.xh]
                    # vx_values=[state[3]]
                    vx_refplot = [state[3] for state in self.x_ref]
                    # jerk=[state[0] for state in self.deltau_record]
                    # deltaf_dot = [state[1] for state in self.deltau_record]
                    jerk = self.jerk_record
                    # = list(itertools.chain.from_iterable(itertools.chain.from_iterable(list3d)))
                    deltaf_dot = self.deltaf_dot_record
                    a = self.action_record[:][0][0]
                    a = [state[0, 0] for state in self.action_record]
                    deltaf = [state[0, 1] for state in self.action_record]

                    def collect_plot_data():
                        """在绘图前调用此方法收集数据"""
                        self.plot_data = {
                            'th': self.th,
                            'Q_record': np.array(self.Q_record),
                            'acc_y_record': self.acc_y_record,
                            'risk_record': self.risk_record,
                            'prob': self.prob,
                            'ed_values': ed_values[:-1],  # 根据你的绘图代码调整
                            'ed_refplot': ed_refplot,
                            'car_speed_record': self.car_speed_record,
                            'vx_refplot': vx_refplot,
                            'jerk': jerk,
                            'deltaf_dot': deltaf_dot,
                            'heading':self.heading_record,
                            'prob1':self.mu_k_1,
                            'prob2':self.mu_k_2,
                            'dis_effect':self.dis_effect_record,
                            'yawrate':yawrate,
                            'actions': {
                                'acc': [a[0, 0] for a in self.action_record],
                                'deltaf': [a[0, 1] for a in self.action_record]
                            },
                            'safety_metrics': {
                                'dis_min': self.dis_min_record,
                                'ttc': self.ttc_record,
                                'lat_ttc': self.lat_ttc_record
                            }
                        }

                    def save_current_data(cbfconfig):
                        """在仿真结束后调用此方法保存数据"""
                        self.data_recorder.save_data(cbfconfig, self.plot_data)
                    collect_plot_data()
                    save_current_data(self.cbfconfig)
                    #Q_record=Q_record[:, :, 0]
                    # Plot vx vs time
                    ax[0, 0].plot(self.th, Q_record[:, 0, 0])
                    ax[0, 0].set_xlabel('T')
                    ax[0, 0].set_ylabel(r'$Q_{ed}$')
                    ax[0, 0].grid(True)

                    ax[0, 1].plot(self.th, self.acc_y_record)
                    ax[0, 1].set_xlabel('T')
                    ax[0, 1].set_ylabel(r'$acc_{y}$')
                    ax[0, 1].grid(True)
                    # ax[0, 1].plot(self.th, self.ay_record)
                    # ax[0, 1].set_xlabel('T')
                    # ax[0, 1].set_ylabel(r'$acc_{y}$')
                    # ax[0, 1].grid(True)
                    ax[1, 0].plot(self.th, self.risk_record)
                    ax[1, 0].set_xlabel('T')
                    ax[1, 0].set_ylabel(r'$risk$')
                    ax[1, 0].grid(True)

                    # ax[1, 1].plot(self.th, self.car_speed_record_sac)
                    # ax[1, 1].set_xlabel('T')
                    # ax[1, 1].set_ylabel(r'$vx$')
                    # ax[1, 1].grid(True)

                    ax[1, 1].plot(self.th, self.prob)
                    ax[1, 1].set_xlabel('T')
                    ax[1, 1].set_ylabel(r'$prob$')
                    ax[1, 1].grid(True)

                    ax[2, 0].plot(self.th, ed_values[:-1])
                    ax[2, 0].set_xlabel('T')
                    ax[2, 0].set_ylabel(r'$ed$')
                    ax[2, 0].grid(True)

                    ax[2, 0].plot(self.th, ed_refplot)
                    ax[2, 0].set_xlabel('T')
                    ax[2, 0].set_ylabel(r'$ed$')
                    ax[2, 0].grid(True)

                    ax[2, 1].plot(self.th, self.car_speed_record)
                    ax[2, 1].set_xlabel('T')
                    ax[2, 1].set_ylabel(r'$vx$')
                    ax[2, 1].grid(True)

                    ax[2, 1].plot(self.th, vx_refplot)
                    ax[2, 1].set_xlabel('T')
                    ax[2, 1].set_ylabel(r'$vx_ref$')
                    ax[2, 1].grid(True)
                    ax[3, 0].plot(self.th, jerk)
                    ax[3, 0].set_xlabel('T')
                    ax[3, 0].set_ylabel(r'$jerk$')
                    ax[3, 0].grid(True)
                    ax[3, 1].plot(self.th, deltaf_dot)
                    ax[3, 1].set_xlabel('T')
                    ax[3, 1].set_ylabel(r'$deltaf_dot$')
                    ax[3, 1].grid(True)
                    ax[4, 0].plot(self.th, a)
                    ax[4, 0].set_xlabel('T')
                    ax[4, 0].set_ylabel(r'$ax$')
                    ax[4, 0].grid(True)
                    ax[4, 1].plot(self.th, deltaf)
                    ax[4, 1].set_xlabel('T')
                    ax[4, 1].set_ylabel(r'$deltaf$')
                    ax[4, 1].grid(True)

                    ax[5, 0].plot(self.th, self.heading_record)
                    ax[5, 0].set_xlabel('T')
                    ax[5, 0].set_ylabel(r'$phi$')
                    ax[5, 0].grid(True)

                    ax[5, 1].plot(self.th, self.dis_min_record)
                    ax[5, 1].set_xlabel('T')
                    ax[5, 1].set_ylabel(r'$dis_min$')
                    ax[5, 1].grid(True)

                    ax[6, 0].plot(self.th, self.mu_k_1[0])
                    ax[6, 0].set_xlabel('T')
                    ax[6, 0].set_ylabel(r'mu_k_1$')
                    ax[6, 0].grid(True)
                    ax[6, 0].plot(self.th, self.mu_k_1[1])
                    ax[6, 0].set_xlabel('T')
                    ax[6, 0].set_ylabel(r'mu_k_1$')
                    ax[6, 0].grid(True)
                    if len(self.mu_k_1[2])==len(self.th):
                        ax[6, 0].plot(self.th, self.mu_k_1[2])
                        ax[6, 0].set_xlabel('T')
                        ax[6, 0].set_ylabel(r'mu_k_1$')
                        ax[6, 0].grid(True)


                    ax[6, 1].plot(self.th, yawrate[:-1])
                    ax[6, 1].set_xlabel('T')
                    ax[6, 1].set_ylabel(r'yawrate$')
                    ax[6, 1].grid(True)
                    ax[7, 0].plot(self.th, self.ttc_record)
                    ax[7, 0].set_xlabel('T')
                    ax[7, 0].set_ylabel(r'ttc')
                    ax[7, 0].grid(True)
                    ax[7, 0].plot(self.th, self.lat_ttc_record)
                    ax[7, 0].set_xlabel('T')
                    ax[7, 0].set_ylabel(r'lat_ttc')
                    ax[7, 0].grid(True)
                    ax[7, 1].plot(self.th, self.dis_effect_record)
                    ax[7, 1].set_xlabel('T')
                    ax[7, 1].set_ylabel(r'dis_effect')
                    ax[7, 1].grid(True)

                    plt.tight_layout()
                    plt.show()
                    print('mean_cost')
                    print(np.mean(cost_value))
                    print('mean_speed')
                    print(np.mean(self.car_speed_record))
                break
        Q_record = (np.array(self.Q_record))
        ed_refplot = self.x_ref[0][0][0]
        ed_values = self.xh[0][0]
        ed_e = ed_values - ed_refplot
        info['ed_e'] = ed_e
        info['jerk'] = self.jerk
        info['delta_f_dot'] = self.delta_f_dot
        info['acc_y']=self.acc_y
        info['dis_min']=dis2obs1
        info['dis_effect']=dis_effect
        info['min_TTC'] = (min(self.ttc_cf,5))
        info['lat_ttc']=lat_ttc_min

        observation = self.observation_gener.generate_observation(current_state=x0, reference_state=self.xs_ori,
                                                                  states=self.states,prob=js_divergence,ada=daptive_para)
        if observation.shape[0]<35:
            print('e')
        #observation=observation.reshape(15,1)
        observation_set.append(observation)

        self.reward_gener = RewardGenerator(observation, self.onroad, self.crash, self.toend, speed,
                                            self.normalization_lims, self.sigmas, reward_env, self.delta_f_dot_min,
                                            self.delta_f_dot_max, self.jerk_min, self.jerk_max, actions,
                                            self.jerk, self.delta_f_dot, self.x0[5],self.xs_ori,dis, self.states_buffer,daptive_para,risk)
        # self.reward_gener = RewardGenerator(observation,self.onroad, self.crash, self.toend,speed)
        # reward=self.reward_gener.generate_reward()
        env_reward, dev_env, self.reward = self.reward_gener.reward_new()
        return observation, self.reward, done, truncated, info

        # Use self.env.step(...) to interact with the underlying Highway-v0 environment

    def predict(self):
        # 筛掉位置为0的障碍物
        non_zero_obstacles = self.obstacles[np.any(self.oobstacles[:, 1:3] != 0, axis=1)]
        obstacles_position = non_zero_obstacles[:, 1:3] + self.oego_position
        # obstacles_vx=(obstacles[:, 3]*cos(obstacles[:, 4])+ego_vx)
        obstacles_vx = (non_zero_obstacles[:, 5])
        # obstacles_vy=obstacles[:, 3]*sin(obstacles[:, 4])+ego_vy
        obstacles_vy = (non_zero_obstacles[:, 6])
        obstacles_ax = (non_zero_obstacles[:, 9])
        obstacles_ay = (non_zero_obstacles[:, 10])
        obstacles_absolute = np.array(
            [obstacles_position[:, 0], obstacles_position[:, 1], obstacles_vx, obstacles_vy, obstacles_ax, obstacles_ay,
             obstacles[:, 11]])
        # obstacles_absolute=np.array(obstacles_absolute).reshape(4,4)
        obstacles_pre = obstacles_absolute
        obstacles_pre = obstacles_pre.T
        # The file 'Model_Parameters.mat' contains the parameters of IAIMM-KF, which is identified offline
        Model_Parameters = loadmat(r'Model_Parameters.mat')
        Model_Parameters = Model_Parameters['Model_Parameters']
        Model_Parameters = Model_Parameters[0, 0]
        m0 = [Model_Parameters['m0']['Lon'][0][0][0], Model_Parameters['m0']['Lat'][0][0][0]]
        m1 = [Model_Parameters['m1']['Lon'][0][0][0], Model_Parameters['m1']['Lat'][0][0][0]]
        m2 = [Model_Parameters['m2']['Lon'][0][0][0], Model_Parameters['m2']['Lat'][0][0][0]]
        m3 = [Model_Parameters['m3']['Lon'][0][0][0], Model_Parameters['m3']['Lat'][0][0][0]]
        m4 = [Model_Parameters['m4']['Lon'][0][0][0], Model_Parameters['m4']['Lat'][0][0][0]]
        m5 = [Model_Parameters['m5']['Lon'][0][0][0], Model_Parameters['m5']['Lat'][0][0][0]]
        m6 = [Model_Parameters['m6']['Lon'][0][0][0], Model_Parameters['m6']['Lat'][0][0][0]]
        std_m0 = [Model_Parameters['m0']['K_set_lon'][0][0][0], Model_Parameters['m0']['std_y'][0][0][0]]
        std_m1 = [Model_Parameters['m1']['K_set_lon'][0][0][0], Model_Parameters['m1']['K_set_lat'][0][0][0]]
        std_m2 = [Model_Parameters['m2']['K_set_lon'][0][0][0], Model_Parameters['m2']['K_set_lat'][0][0][0]]
        std_m3 = [Model_Parameters['m3']['K_set_lon'][0][0][0], Model_Parameters['m3']['std_y'][0][0][0]]
        std_m4 = [Model_Parameters['m4']['K_set_lon'][0][0][0], Model_Parameters['m4']['K_set_lat'][0][0][0]]
        std_m5 = [Model_Parameters['m5']['K_set_lon'][0][0][0], Model_Parameters['m5']['K_set_lat'][0][0][0]]
        std_m6 = [Model_Parameters['m6']['K_set_lon'][0][0][0], Model_Parameters['m6']['std_y'][0][0][0]]
        Models = [m0, m1, m2, m3, m4, m5, m6]  # submodels (controller gains of nominal maneuvers)
        std_parameters = [std_m0, std_m1, std_m2, std_m3, std_m4, std_m5, std_m6]  # parameters of standard deviation
        # basic parameters
        Ts = 0.05
        N = 20
        N_Lane = 3
        N_M = 7  # 后面看以下参数修改一下模态数量
        N_M_EV = 3  # 应该没啥用
        N_Car = 1
        index_EV = 0
        L_Width = [4.0, 4.0, 4.0]
        L_Bound = [-2, 2, 6, 10]
        L_Center = [0 / 2, 2 + 4.0 / 2, 6 + 4.0 / 2]
        l_veh = 5.0
        w_veh = 2.04
        DSV = 6
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        infinity = 100000
        Q = np.diag([1, 0.5, 0.25, 0.1, 0.1, 0])
        R = np.diag([1, 1, 1]) * 1e-5
        miuR = np.array([0, 0, 0])
        K_sampling = 30
        SpeedLim = np.array([None, None, None])  ##这个也没懂
        Weight = np.array([0.2, 0.1, 0.5, 0.2])  ##没看懂这个是干嘛的
        ### This is a very important parameter
        epsilon = 0.1  # the safety parameter of MPC (epsilon = 1 is deterministic MPC)
        opts_SV = {
            'Ts': Ts,
            'N': N,
            'N_Lane': N_Lane,
            'N_M': N_M,
            'N_Car': N_Car,
            'L_Width': L_Width,
            'w_veh': w_veh,
            'l_veh': l_veh,
            'L_Bound': L_Bound,
            'L_Center': L_Center,
            'DSV': DSV,
            'infinity': infinity,
            'SpeedLim': SpeedLim,
            'Q': Q,
            'R': R,
            'Weight': Weight,
            'H': H,
            'Models': Models,
            'std_parameters': std_parameters,
            'K_sampling': K_sampling
        }

        # IMM_KF = IAIMM_KF(Params=opts_SV)
        if self.step == 1:
            X_state_0 = []

            i = 0
            for obs in obstacles:
                veh_id = obs[6]

                x_0 = np.array([obs[0], obs[2], obs[4], obs[1], obs[3], obs[5]])

                # X_state_0.append([x_0])
                self.state_manegers[i].X_State.append([x_0])
                self.state_manegers[i].tracked_vehicles.append([veh_id])
                Initial_SV = Initialization_SV(Params=opts_SV)
                MU_0, M_0, Y_0, X_Hat_0, P_0, X_Pre_0, X_Po_All_0, X_Var_0, Y_Var_0, REF_Speed_0, REF_Lane_0, REF_Speed_All_0 = Initial_SV.Initialize_MU_M_P(
                    self.state_manegers[i].X_State)
                self.state_manegers[i].MU.append(MU_0)
                self.state_manegers[i].M.append(M_0)
                self.state_manegers[i].Y.append(Y_0)
                self.state_manegers[i].X_Hat.append(X_Hat_0)
                self.state_manegers[i].X_pre.append(X_Pre_0)
                self.state_manegers[i].X_Po_All.append(X_Po_All_0)
                self.state_manegers[i].X_Var.append(X_Var_0)
                self.state_manegers[i].Y_Var.append(Y_Var_0)
                self.state_manegers[i].P.append(P_0)
                self.state_manegers[i].Ref_Speed.append(REF_Speed_0)
                self.state_manegers[i].Ref_Lane.append(REF_Lane_0)
                self.state_manegers[i].Ref_Speed_All.append(REF_Speed_All_0)
                i = i + 1
        elif self.step > 1:
            Y_k = []
            new_tracked_vehicles = []
            update_X_state = []
            new_state_managers = []
            for obs in obstacles_pre:
                veh_id = obs[6]
                y_k = np.array([obs[0], obs[2], obs[1]])
                found = False
                for state_manager in self.state_manegers:
                    if veh_id in state_manager.tracked_vehicles:
                        idx = state_manager.tracked_vehicles.index(veh_id)
                        state_manager.Y.append(y_k)
                        new_state_managers.append(state_manager)
                        found = True
                        break
                if not found:
                    x_0 = np.a4([obs[0], obs[2], obs[4], obs[1], obs[3], obs[5]])
                    new_state_manager = StateManager()
                    new_state_manager.tracked_vehicles.append(veh_id)
                    new_state_manager.X_State.append(x_0)
                    Initial_SV = Initialization_SV(Params=opts_SV)
                    MU_0, M_0, Y_0, X_Hat_0, P_0, X_Pre_0, X_Po_All_0, X_Var_0, Y_Var_0, REF_Speed_0, REF_Lane_0, REF_Speed_All_0 = Initial_SV.Initialize_MU_M_P(
                        new_state_manager.X_State)
                    new_state_manager.MU.append(MU_0)
                    new_state_manager.M.append(M_0)
                    new_state_manager.Y.append(Y_0)
                    new_state_manager.X_Hat.append(X_Hat_0)
                    new_state_manager.X_pre.append(X_Pre_0)
                    new_state_manager.X_Po_All.append(X_Po_All_0)
                    new_state_manager.X_Var.append(X_Var_0)
                    new_state_manager.Y_Var.append(Y_Var_0)
                    new_state_manager.P.append(P_0)
                    new_state_manager.Ref_Speed.append(REF_Speed_0)
                    new_state_manager.Ref_Lane.append(REF_Lane_0)
                    new_state_manager.Ref_Speed_All.append(REF_Speed_All_0)
                    new_state_managers.append(new_state_manager)
            self.state_manegers = new_state_managers

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.closestpoint_index_old = 0
        self.steps = 0
        self.state_maneger1 = StateManager()
        self.state_maneger2 = StateManager()
        self.state_maneger3 = StateManager()
        self.state_maneger4 = StateManager()
        # self.state_maneger5 = StateManager()
        self.state_manegers = [self.state_maneger1, self.state_maneger2, self.state_maneger3, self.state_maneger4]
        self.mu_k_1 = [[], [], []]
        self.mu_k_2 = [[], [], []]
        self.mu_k_3 = [[], [], []]
        self.states = self.env.reset()
        self.observation_position = self.states[0][0][1:3]
        self.ego_position = self.observation_position
        self.ego_v = self.states[0][0][3]
        self.lookahead = 10
        self.safe_distance = 15
        self.phi = self.states[0][0][7]
        self.v_x = self.states[0][0][5]
        self.v_y = self.states[0][0][6]
        self.V = sqrt(self.v_x ** 2 + self.v_y ** 2)
        self.r = self.states[0][0][8]
        self.delta_f = 0
        self.danger_time = 0
        self.reward = 0
        self.risk_time=0
        self.p = 0
        self.x0 = np.array(
            [self.ego_position[1], np.arctan2(np.sin(self.phi), np.cos(self.phi)), self.ego_position[0], self.v_x,
             self.v_y, self.r]).reshape(-1, 1)
        if self.x0[0]>6:
            self.ed_initial=8
        elif self.x0[0]>2:
            self.ed_initial=4
        else:
            self.ed_initial=0
        self.lane_change = 0
        self.x_list = []
        self.y_list = []
        self.vx_list = []
        self.vy_list = []

        obs = self.states
        self.xs = self.mpc_solver.xs
        self.xs = np.tile(self.xs, (1, 21))
        #observation = [obs[0][0][2] - self.trajectory_ref[1][0], obs[0][0][3] - self.trajectory_ref[4][0]]
        observation = np.zeros(35)
        info = {}


        return observation, info
        # ... (reset logic using self.env.reset()) ...

    def updateReference(self, r=np.zeros((4, 1))):
        """
        Updates the y position reference for each controller based on the current lane
        """

        # 获取当前车辆位置
        py_ego = self.ego_position[1]
        self.egoPx = self.ego_position[0]
        # print("INFO: Ego position x Measurement is:", self.egoPx)
        # print("INFO: Ego position y Measurement is:", py_ego)
        # 初始化参考输入
        refu_in = [0, 0, 0]
        # 获取参考值
        refxT_in = self.ed_refT[0]
        refxR_in = self.ed_refC[0]

        # 车道中心
        left_lane_center = 4.0  # 左车道中心
        right_lane_center = 0.0  # 右车道中心
        tol = 0.2  # 容差
        # 根据车辆位置设置参考值
        if py_ego >= left_lane_center - 0.5:
            # 如果车辆接近右车道
            refxT_in = left_lane_center  # 主参考设为右车道
            refxR_in = right_lane_center  # 左参考设为左车道
            self.ed_refC = self.getReference(refxR_in)
            print("INFO: Vehicle is in or near the right lane!")
        elif py_ego < right_lane_center + 0.5:
            # 如果车辆接近左车道
            refxT_in = right_lane_center  # 主参考设为左车道
            refxL_in = left_lane_center  # 右参考设为右车道
            self.ed_refC = self.getReference(refxL_in)
            print("INFO: Vehicle is in or near the left lane!")
            # Trailing reference should always be the current Lane!

        # 获取最终的参考值
        self.ed_refT = self.getReference(refxT_in)
        self.ed_refC = self.getReference(abs(4 - refxT_in))

    def getReference(self, ref_in):
        ref_out = np.tile(ref_in, self.N + 1)
        return ref_out

    def save_trajectory(self, filename_prefix):
        filename = f"{filename_prefix}_{self.filecounter}.json"
        trajectory_data = {
            'x': self.x_list,
            'y': self.y_list,
            'vx': self.vx_list,
            'vy': self.vy_list
        }
        try:
            with open(filename, 'w') as f:
                json.dump(trajectory_data, f, indent=4)
            print(f"文件已成功保存到: {filename}")
            self.filecounter += 1
        except Exception as e:
            print(f"保存文件时出错: {e}")
    def save_trajectory_sv(self, filename_prefix):
        filename = f"{filename_prefix}_{self.filecounter}.json"
        trajectory_data = {
            'x': self.x_svlist,
            'y': self.y_svlist,
            'vx': self.vx_svlist,
            'vy': self.vy_svlist
        }

        try:
            with open(filename, 'w') as f:
                json.dump(trajectory_data, f, indent=4)
            print(f"文件已成功保存到: {filename}")
            self.filecounter += 1
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def save_trajectory_sv0(self, filename_prefix):
        filename = f"{filename_prefix}_{self.filecounter}.json"
        trajectory_data = {
            'x': self.x_sv0list,
            'y': self.y_sv0list,
            'vx': self.vx_sv0list,
            'vy': self.vy_sv0list
        }

        try:
            with open(filename, 'w') as f:
                json.dump(trajectory_data, f, indent=4)
            print(f"文件已成功保存到: {filename}")
            self.filecounter += 1
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def save_trajectory_sv1(self, filename_prefix):
        filename = f"{filename_prefix}_{self.filecounter}.json"
        trajectory_data = {
            'x': self.x_sv1list,
            'y': self.y_sv1list,
            'vx': self.vx_sv1list,
            'vy': self.vy_sv1list
        }

        try:
            with open(filename, 'w') as f:
                json.dump(trajectory_data, f, indent=4)
            print(f"文件已成功保存到: {filename}")
            self.filecounter += 1
        except Exception as e:
            print(f"保存文件时出错: {e}")
    # ... (other methods for rendering, etc.) ...

    def jensen_shannon_divergence(self,p, q):
        """
        计算两个概率分布 p 和 q 的 Jensen-Shannon 散度
        参数:
            p: 第一个概率分布，数组形式 (如 [0.7, 0.2, 0.1])
            q: 第二个概率分布，数组形式 (如 [1/3, 1/3, 1/3])
        返回:
            JS散度值
        """
        # 转换为 numpy 数组，确保计算正确
        p = np.array(p)
        q = np.array(q)
        # 计算平均分布 M
        m = 0.5 * (p + q)
        # 计算 KL 散度
        kl_p_m = entropy(p, m)  # KL(p || m)
        kl_q_m = entropy(q, m)  # KL(q || m)
        # 计算 JS 散度
        js_divergence = 0.5 * (kl_p_m + kl_q_m)
        return js_divergence

