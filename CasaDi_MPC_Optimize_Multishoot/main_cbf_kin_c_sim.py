# import CBF_MPC_Solver
import MPC_CBF_optimize_kin
import RefPathGenerator
import helpers
from helpers import load_config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import time
import casadi as ca
from Obs_prediction import obs_prediction

PARAMS_FILE = "mpc_parameters.yaml"

def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[0, :])
    st = x0 + T*f_value.full()
    t = t0 + T
    # print(u[:,0])
    # u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    u_end = np.concatenate((u[1:], u[-1:]))
    # x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[1:], x_f[-1:]), axis=0)

    return t, st, u_end, x_f

if __name__ == "__main__":
    config = load_config(PARAMS_FILE)
    mpc_params = config['mpc_params']
    T_horizon = mpc_params['horizon']
    T_S = mpc_params['T_S']
    t_vector = np.arange(0, T_horizon+T_S, T_S, dtype=float)
    N_p = len(t_vector) - 1
    print("----main N_p: ", N_p)


    # Simulation
    # initialize
    mpc_solver = MPC_CBF_optimize_kin.MPC_optimize()
    n_states = mpc_solver.num_states
    n_controls = mpc_solver.num_controls
    t0 = 0.0
    # x y phi vx
    x0 = np.array([0, 3, 0, 15]).reshape(-1, 1)# initial state
    x0_ = x0.copy()
    x_m = np.zeros((n_states, N_p+1))
    next_states = x_m.copy().T
    xs = np.array([400, 3.5, 0, 30]).reshape(-1, 1) # final state
    u0 = np.array([0, 0]*N_p).reshape(-1, 2).T  # np.ones((N_p, 2)) # controls
    ref = RefPathGenerator.RefPathGenerator()
    ref_global = ref.define_ref_path(x0, xs, T_S)
    # obs = np.array([[50, 3.5], [70, 0.0],[100, 3.5]]) # 障碍物位置
    # obs = np.array([[50, 3.5, 4.8, 1.8], [70, 0.0, 4.8, 1.8],[100, 3.5, 4.8, 1.8]]) # 障碍物位置+长宽
    obs = np.array([[50, 3.5, 0, 8, 4.8, 1.8]]) # 障碍物位置+长宽
    # obs_trajectories = obs_prediction(obs, T_S, N_p+1)
    obs_trajectories = obs
    xsq = []
    usq = []
    usq0 = []
    refsq = []
    u_exc = []
    xh = [] # contains for the history of the state
    uh = []
    th = [] # for the time
    traj = []
    caltimeh = [0]
    sim_time = 8
    T_sim=0.1
    

    # start MPC
    mpciter = 0
    index_t = []
    # initialize MPC solver
    # mpc_solver = MPC_optimize_kin.MPC_optimize()
    lbg, ubg, lbx, ubx = mpc_solver.initialize_constraints(obs)

    solve_flag = True
    plot_opt_sq_flag = True
    plot_sim_flag = True
    # solve_flag = False
    # plot_flag = False
    executate_max = 1
    last_idx = 0
    if solve_flag:
        while( mpciter-sim_time/T_S < 0.0):
            start_time = time.time()
            # set parameter
            c_p = np.concatenate((x0, xs))
            # c_p = np.concatenate((c_p, obs))
            init_control = np.concatenate((u0.reshape(-1, 1), next_states.reshape(-1, 1)))
            # ref_traj = mpc_solver.generate_ref_path(x0, xs)   # 多向式
            # ref_traj = np.array([400, 3.5, 0, 25]*(N_p+1)).reshape(-1, 4)# 参考车道中心线
            # x_ref = np.linspace(0, xs[3][0]*T_horizon, N_p+1).reshape(-1, 1)  # 确保x_ref是二维数组
            # ref_traj[:, 0] = x_ref.ravel()  # 使用ravel()将x_ref展平为一维数组
            # 从global refpath中找最近index，生成N_p+1个ref_traj point
            ref_traj, last_idx = ref.find_ref_traj(x0, xs, T_horizon, T_S, last_idx)
            solver = mpc_solver.optimize_problem(ego_state=x0, ref_state=ref_traj, obstacle=obs_trajectories)
            res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
            # the feedback is in the series [u0, x0, u1, x1, ...]
            solve_opt = res['x'].full()
            u0 = solve_opt[:N_p*n_controls].reshape(N_p, n_controls)  # 前N_p*n_controls个值 -> (N_p, n_controls)
            x_m = solve_opt[N_p*n_controls:].reshape(N_p+1, n_states)  # 后N_p+1*n_states个值 -> (N_p+1, n_states)
            if mpciter == 1:
            # if x_m[0, 0] > 25:
                xsq.append(x_m.T)
                usq.append(u0.T)
                usq0.append(u0[0, :])
                u_exc.append(u0[:executate_max, :].T)
                refsq.append(ref_traj)
            
            execiter = 0
            while execiter < executate_max:
                uh.append(u0[0, :]) #添加当前控制量
                th.append(t0)   
                t0, x0, u0, next_states = shift_movement(T_S, t0, x0, u0, x_m, mpc_solver.f) #根据当前控制量和状态量递推下一时刻
                x0 = ca.reshape(x0, -1, 1)
                x0 = x0.full()
                xh.append(x0)
                execiter = execiter + 1
            mpciter = mpciter + executate_max
            caltimeh.append((time.time() - start_time)*1000)

    if plot_opt_sq_flag:
        x_values = xsq[0][0,:]
        y_values = xsq[0][1,:]
        phi_values = xsq[0][2,:] * 180 / np.pi
        vx_values = xsq[0][3,:]
        df_values = usq[0][0,:] * 180 / np.pi
        ax_values = usq[0][1,:]
        time_values = t_vector

        fig0 = plt.figure(figsize=(10,5))
        fig0.suptitle("opt_sq0", fontsize=16)
        # Plot x vs time
        axx = plt.subplot(3, 3, 1)
        axx.plot(time_values, x_values)
        axx.set_xlabel('T')
        axx.set_ylabel(r'$X$')
        axx.grid(True)
        ref_x = refsq[0][:,0]
        axx.plot(time_values, ref_x, 'r--')

        # Plot y vs time
        axy = plt.subplot(3, 3, 4)
        axy.plot(time_values, y_values)
        axy.set_xlabel('T(s)')
        axy.set_ylabel(r'$Y$')
        axy.grid(True)
        ref_y = refsq[0][:,1]
        axy.plot(time_values, ref_y, 'r--')
        
        # Plot phi vs time
        axp = plt.subplot(3, 3, 2)
        axp.plot(time_values, phi_values)
        axp.set_xlabel('T(s)')
        axp.set_ylabel(r'$phi$')
        axp.set_ylim([-10, 10])
        axp.grid(True)
        ref_phi = refsq[0][:,2]
        axp.plot(time_values, ref_phi, 'r--')


        # Plot vx vs time
        axv = plt.subplot(3, 3, 5)
        axv.plot(time_values, vx_values)
        axv.set_xlabel('T(s)')
        axv.set_ylabel(r'$v_{x}$')
        axv.set_ylim([0, 40])
        axv.grid(True)
        ref_v = refsq[0][:,3]
        axv.plot(time_values, ref_v, 'r--')

        # Plot df vs time
        axd =  plt.subplot(3, 3, 3)
        axd.plot(time_values[:-1], df_values)
        axd.set_xlabel('T')
        axd.set_ylabel(r'$\delta_{f}$')
        axd.set_ylim([-10, 10])
        axd.grid(True)
        axd.text(x=1.5, y=4, s='pandf0 =%.3f' % df_values[0], 
         fontsize=10, color='blue', ha='center')
        real_df = usq0[0][0]*180/np.pi
        axd.text(x=1.5, y=6, s='realdf0 =%.3f' % real_df, 
         fontsize=10, color='blue', ha='center')
        u_exc_df = u_exc[0][0, :]*180/np.pi
        axd.plot(time_values[:executate_max], u_exc_df, 'r--')

        # Plot ax vs time
        axa =  plt.subplot(3, 3, 6)
        axa.plot(time_values[:-1],ax_values)
        axa.set_xlabel('T(s)')
        axa.set_ylabel(r'$a_{x}$')
        axa.set_ylim([-5, 5])
        axa.grid(True)
        u_exc_ax = u_exc[0][1, :]
        axa.plot(time_values[:executate_max], u_exc_ax, 'r--')

        # Plot x vs y
        axxy =  plt.subplot(3, 1, 3)
        axxy.plot(x_values,y_values)
        axxy.set_xlabel('X')
        axxy.set_ylabel(r'$Y$')
        axxy.grid(True)
        for i in range(obs.shape[0]):
            # xy = tuple(obs[i])
            xy = tuple(obs[i][:2])
            width = obs[i][4]
            height = obs[i][5]
            ellipse = Ellipse(xy=xy, width=width, height=height, 
                        angle=0, color='b', fill=False)  # angle表示旋转角度(度)
            axxy.add_patch(ellipse)  # 添加到子图中
        axxy.set_aspect(2)
        ref_x = refsq[0][:,0]
        ref_y = refsq[0][:,1]
        axxy.plot(ref_x, ref_y, 'r--')

    if plot_sim_flag:
        time_values = th

        x_values = [state[0] for state in xh]
        y_values = [state[1] for state in xh]
        phi_values = [state[2] * 180 / np.pi for state in xh]
        vx_values = [state[3] for state in xh]

        df_values = [control[0] * 180 / np.pi for control in uh] 
        ax_values = [control[1] for control in uh]

        cal_time_values = caltimeh

        fig1 = plt.figure(figsize=(10, 5))
        fig1.suptitle("sim result", fontsize=16)
        # Plot x vs time
        axx =  plt.subplot(3, 3, 1)
        axx.plot(time_values, x_values)
        axx.set_xlabel('T')
        axx.set_ylabel(r'$X$')
        axx.grid(True)

        # Plot y vs time
        axy =  plt.subplot(3, 3, 4)
        axy.plot(time_values, y_values)
        axy.set_xlabel('T(s)')
        axy.set_ylabel(r'$Y$')
        axy.grid(True)

        # Plot phi vs time
        axp =  plt.subplot(3, 3, 2)
        axp.plot(time_values, phi_values)
        axp.set_xlabel('T(s)')
        axp.set_ylabel(r'$phi$')
        axp.set_ylim([-10, 10])
        axp.grid(True)

        # Plot vx vs time
        axv =  plt.subplot(3, 3, 5)
        axv.plot(time_values, vx_values)
        axv.set_xlabel('T(s)')
        axv.set_ylabel(r'$v_{x}$')
        axv.set_ylim([0, 40])
        axv.grid(True)

        # Plot df vs time
        axd =  plt.subplot(3, 3, 3)
        axd.plot(time_values, df_values)
        axd.set_xlabel('T')
        axd.set_ylabel(r'$\delta_{f}$')
        axd.set_ylim([-15, 15])
        axd.grid(True)

        # Plot ax vs time
        axa =  plt.subplot(3, 3, 6)
        axa.plot(time_values,ax_values)
        axa.set_xlabel('T(s)')
        axa.set_ylabel(r'$a_{x}$')
        axa.set_ylim([-5, 5])
        axa.grid(True)

        axxy =  plt.subplot(3, 1, 3)    
        axxy.plot(x_values,y_values)
        axxy.set_xlabel('X')
        axxy.set_ylabel(r'$Y$')
        axxy.grid(True)
        # width和height分别表示椭圆的宽度和高度（直径长度）
        for i in range(obs.shape[0]):
            # xy = tuple(obs[i])
            xy = tuple(obs[i][:2])
            width = obs[i][4]
            height = obs[i][5]
            ellipse = Ellipse(xy=xy, width=width, height=height, 
                        angle=0, color='b', fill=False)  # angle表示旋转角度(度)
            axxy.add_patch(ellipse)  # 添加到子图中
        len = int(0.5*ref_global.shape[0])
        axxy.plot(ref_global[:len,0], ref_global[:len,1], 'r--')
        axxy.set_aspect(4)

        # ax[1, 3].plot(time_values,caltimeh[:-1])
        # ax[1, 3].set_xlabel('T(s)')
        # ax[1, 3].set_ylabel(r'$cal_time (ms)$')
        # ax[1, 3].grid(True)


        # Adjust layout and show the plot
    if plot_opt_sq_flag or plot_sim_flag:
        plt.tight_layout()
        plt.show()
