# import CBF_MPC_Solver
import MPC_CBF_optimize_dyn
import helpers
from helpers import load_config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import time
import casadi as ca


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
    mpc_solver = MPC_CBF_optimize_dyn.MPC_optimize()
    n_states = mpc_solver.num_states
    n_controls = mpc_solver.num_controls
    t0 = 0.0
    # x y phi vx vy r
    x0 = np.array([0, 0, 0, 10, 0, 0]).reshape(-1, 1)# initial state
    x0_ = x0.copy()
    x_m = np.zeros((n_states, N_p+1))
    next_states = x_m.copy().T
    xs = np.array([600, 3.5, 0, 15, 0, 0]).reshape(-1, 1) # final state
    u0 = np.array([0, 0]*N_p).reshape(-1, 2).T  # np.ones((N_p, 2)) # controls
    # obs = np.array([100, 3.5]).reshape(-1, 1)  # 障碍物位置
    obs = np.array([100, -3.5]) # 障碍物位置
    xsq = []
    usq = []
    xh = [] # contains for the history of the state
    uh = []
    th = [] # for the time
    traj = []
    caltimeh = [0]
    sim_time = 10
    T_sim=0.1

    # start MPC
    mpciter = 0
    index_t = []
    # initialize MPC solver
    # mpc_solver = MPC_optimize_kin.MPC_optimize()
    lbg, ubg, lbx, ubx = mpc_solver.initialize_constraints()

    solve_flag = True
    plot_opt_sq_flag = True
    plot_sim_flag = True
    # solve_flag = False
    # plot_flag = False
    if solve_flag:
        while( mpciter-sim_time/T_S < 0.0):
            start_time = time.time()
            # set parameter
            c_p = np.concatenate((x0, xs))
            # c_p = np.concatenate((c_p, obs))
            init_control = np.concatenate((u0.reshape(-1, 1), next_states.reshape(-1, 1)))
            # for i in range(1,N_p+1):#0-14
            #     # xs = np.array([30, 0, 0, 0, X_g_ref[i], 1.75]).reshape(-1, 1)
            #     xs = np.array([100, 1.75, 0, 25]).reshape(-1, 1)
            #     print("xs: ", xs)
            #     if i == 1:
            #         c_p = np.concatenate((x0, xs),axis=1)
            #     else:
            #         c_p = np.concatenate((c_p, xs),axis=1)
            solver = mpc_solver.optimize_problem(ego_state=x0, ref_state=xs, obstacle=obs)
            res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
            # the feedback is in the series [u0, x0, u1, x1, ...]
            solve_opt = res['x'].full()
            u0 = solve_opt[:N_p*n_controls].reshape(N_p, n_controls)  # 前N_p*n_controls个值 -> (N_p, n_controls)
            x_m = solve_opt[N_p*n_controls:].reshape(N_p+1, n_states)  # 后N_p+1*n_states个值 -> (N_p+1, n_states)
            if mpciter == 0:
                xsq.append(x_m.T)
                usq.append(u0.T)
            if mpciter == 10:
                u0[0, 0] = 0
                u0[0, 1] = 0
            uh.append(u0[0, :]) #添加当前控制量
            th.append(t0)
            t0, x0, u0, next_states = shift_movement(T_S, t0, x0, u0, x_m, mpc_solver.f) #根据当前控制量和状态量递推下一时刻
            x0 = ca.reshape(x0, -1, 1)
            x0 = x0.full()
            xh.append(x0)
            mpciter = mpciter + 1
            caltimeh.append((time.time() - start_time)*1000)

    if plot_opt_sq_flag:
        x_values = xsq[0][0,:]
        y_values = xsq[0][1,:]
        phi_values = xsq[0][2,:]
        vx_values = xsq[0][3,:]
        df_values = usq[0][0,:] * 180 / np.pi
        ax_values = usq[0][1,:]
        time_values = t_vector

        fig0 = plt.figure(figsize=(12, 6))

        # Plot x vs time
        axx = plt.subplot(3, 3, 1)
        axx.plot(time_values, x_values)
        axx.set_xlabel('T')
        axx.set_ylabel(r'$X$')
        axx.grid(True)

        # Plot y vs time
        axy = plt.subplot(3, 3, 4)
        axy.plot(time_values, y_values)
        axy.set_xlabel('T(s)')
        axy.set_ylabel(r'$Y$')
        axy.grid(True)
        
        # Plot phi vs time
        axp = plt.subplot(3, 3, 2)
        axp.plot(time_values, phi_values)
        axp.set_xlabel('T(s)')
        axp.set_ylabel(r'$phi$')
        axp.set_ylim([-10, 10])
        axp.grid(True)

        # Plot vx vs time
        axv = plt.subplot(3, 3, 5)
        axv.plot(time_values, vx_values)
        axv.set_xlabel('T(s)')
        axv.set_ylabel(r'$v_{x}$')
        axv.set_ylim([0, 40])
        axv.grid(True)

        # Plot df vs time
        axd =  plt.subplot(3, 3, 3)
        axd.plot(time_values[:-1], df_values)
        axd.set_xlabel('T')
        axd.set_ylabel(r'$\delta_{f}$')
        axd.set_ylim([-15, 15])
        axd.grid(True)

        # Plot ax vs time
        axa =  plt.subplot(3, 3, 6)
        axa.plot(time_values[:-1],ax_values)
        axa.set_xlabel('T(s)')
        axa.set_ylabel(r'$a_{x}$')
        axa.set_ylim([-5, 5])
        axa.grid(True)
        # Plot x vs y
        axxy =  plt.subplot(3, 1, 3)
        axxy.plot(x_values,y_values)
        axxy.set_xlabel('X')
        axxy.set_ylabel(r'$Y$')
        axxy.grid(True)
        ellipse = Ellipse((obs  [0], obs  [1]), width=2, height=1, 
                    angle=0, color='b', fill=False)  # angle表示旋转角度(度)
        axxy.add_patch(ellipse)  # 添加到子图中
        axxy.set_aspect(1)
      
    if plot_sim_flag:
        time_values = th

        x_values = [state[0] for state in xh]
        y_values = [state[1] for state in xh]
        phi_values = [state[2] for state in xh]
        vx_values = [state[3] for state in xh]

        df_values = [control[0]* 180 / np.pi for control in uh] 
        ax_values = [control[1] for control in uh]

        cal_time_values = caltimeh

        fig1, ax1 = plt.subplots(2, 3, figsize=(14, 6))
        # Plot x vs time
        ax1[0, 0].plot(time_values, x_values)
        ax1[0, 0].set_xlabel('T')
        ax1[0, 0].set_ylabel(r'$X$')
        ax1[0, 0].grid(True)

        # Plot y vs time
        ax1[1, 0].plot(time_values, y_values)
        ax1[1, 0].set_xlabel('T(s)')
        ax1[1, 0].set_ylabel(r'$Y$')
        ax1[1, 0].grid(True)

        # Plot phi vs time
        ax1[0, 1].plot(time_values, phi_values)
        ax1[0, 1].set_xlabel('T(s)')
        ax1[0, 1].set_ylabel(r'$phi$')
        ax1[0, 1].set_ylim([-10, 10])
        ax1[0, 1].grid(True)

        # Plot vx vs time
        ax1[1, 1].plot(time_values, vx_values)
        ax1[1, 1].set_xlabel('T(s)')
        ax1[1, 1].set_ylabel(r'$v_{x}$')
        ax1[1, 1].set_ylim([0, 40])
        ax1[1, 1].grid(True)

        # Plot df vs time
        ax1[0, 2].plot(time_values, df_values)
        ax1[0, 2].set_xlabel('T')
        ax1[0, 2].set_ylabel(r'$\delta_{f}$')
        ax1[0, 2].set_ylim([-15, 15])
        ax1[0, 2].grid(True)

        # Plot ax vs time
        ax1[1, 2].plot(time_values,ax_values)
        ax1[1, 2].set_xlabel('T(s)')
        ax1[1, 2].set_ylabel(r'$a_{x}$')
        ax1[1, 2].set_ylim([-5, 5])
        ax1[1, 2].grid(True)

        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 3))

        ax2.plot(x_values,y_values)
        ax2.set_xlabel('X')
        ax2.set_ylabel(r'$Y$')
        ax2.grid(True)
        # width和height分别表示椭圆的宽度和高度（直径长度）
        ellipse = Ellipse((obs  [0], obs  [1]), width=2, height=1, 
                    angle=0, color='b', fill=False)  # angle表示旋转角度(度)
        ax2.add_patch(ellipse)  # 添加到子图中
        ax2.set_aspect(1)

        # ax[1, 3].plot(time_values,caltimeh[:-1])
        # ax[1, 3].set_xlabel('T(s)')
        # ax[1, 3].set_ylabel(r'$cal_time (ms)$')
        # ax[1, 3].grid(True)


        # Adjust layout and show the plot
    if plot_opt_sq_flag or plot_sim_flag:
        plt.tight_layout()
        plt.show()
