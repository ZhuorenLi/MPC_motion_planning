# import CBF_MPC_Solver
import MPC_optimize_kin
import helpers
from helpers import load_config
import numpy as np
import matplotlib.pyplot as plt
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
    mpc_solver = MPC_optimize_kin.MPC_optimize()
    n_states = mpc_solver.num_states
    n_controls = mpc_solver.num_controls
    t0 = 0.0
    # x y phi vx
    x0 = np.array([0, 0, 0, 20]).reshape(-1, 1)# initial state
    x0_ = x0.copy()
    x_m = np.zeros((n_states, N_p+1))
    next_states = x_m.copy().T
    xs = np.array([500, 3.5, 0, 30]).reshape(-1, 1) # final state
    u0 = np.array([0, 0]*N_p).reshape(-1, 2).T  # np.ones((N_p, 2)) # controls
    
    xc = []
    uc = []
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
    plot_flag = True
    # solve_flag = False
    # plot_flag = False
    if solve_flag:

        start_time = time.time()
        # set parameter
        c_p = np.concatenate((x0, xs))
        init_control = np.concatenate((u0.reshape(-1, 1), next_states.reshape(-1, 1)))
        # for i in range(1,N_p+1):#0-14
        #     # xs = np.array([30, 0, 0, 0, X_g_ref[i], 1.75]).reshape(-1, 1)
        #     xs = np.array([100, 1.75, 0, 25]).reshape(-1, 1)
        #     print("xs: ", xs)
        #     if i == 1:
        #         c_p = np.concatenate((x0, xs),axis=1)
        #     else:
        #         c_p = np.concatenate((c_p, xs),axis=1)
        solver = mpc_solver.optimize_problem(ego_state=x0, ref_state=xs)
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        # the feedback is in the series [u0, x0, u1, x1, ...]
        solve_opt = res['x'].full()
        u0 = solve_opt[:N_p*n_controls].reshape(N_p, n_controls)  # 前N_p*n_controls个值 -> (N_p, n_controls)
        x_m = solve_opt[N_p*n_controls:].reshape(N_p+1, n_states)  # 后N_p+1*n_states个值 -> (N_p+1, n_states)
        xc.append(x_m.T)
        uc.append(u0.T)
        uh.append(u0[0, :])
        th.append(t0)
        t0, x0, u0, next_states = shift_movement(T_S, t0, x0, u0, x_m, mpc_solver.f)
        x0 = ca.reshape(x0, -1, 1)
        x0 = x0.full()
        xh.append(x0)
        mpciter = mpciter + 1
        caltimeh.append((time.time() - start_time)*1000)
    
    if plot_flag:
        time_values = t_vector

        # x_values = np.array([state[0] for state in xc]).flatten()
        # y_values = np.array([state[1] for state in xc]).flatten()
        # phi_values = np.array([state[2] for state in xc]).flatten()
        # vx_values = np.array([state[3] for state in xc]).flatten()

        # df_values = np.array([control[0]* 180 / np.pi for control in uc]).flatten() 
        # ax_values = np.array([control[1] for control in uh]).flatten()
        x_values = xc[0][0,:]
        y_values = xc[0][1,:]
        phi_values = xc[0][2,:]
        vx_values = xc[0][3,:]
        df_values = uc[0][0,:]
        ax_values = uc[0][1,:]
        
        fig, ax = plt.subplots(2, 4, figsize=(16, 6))
        # Plot x vs time
        ax[0, 0].plot(time_values, x_values)
        ax[0, 0].set_xlabel('T')
        ax[0, 0].set_ylabel(r'$X$')
        ax[0, 0].grid(True)

        # Plot y vs time
        ax[1, 0].plot(time_values, y_values)
        ax[1, 0].set_xlabel('T(s)')
        ax[1, 0].set_ylabel(r'$Y$')
        ax[1, 0].grid(True)

        # Plot phi vs time
        ax[0, 1].plot(time_values, phi_values)
        ax[0, 1].set_xlabel('T(s)')
        ax[0, 1].set_ylabel(r'$phi$')
        ax[0, 1].set_ylim([-10, 10])
        ax[0, 1].grid(True)

        # Plot vx vs time
        ax[1, 1].plot(time_values, vx_values)
        ax[1, 1].set_xlabel('T(s)')
        ax[1, 1].set_ylabel(r'$v_{x}$')
        ax[1, 1].set_ylim([0, 40])
        ax[1, 1].grid(True)

        # Plot df vs time
        ax[0, 2].plot(time_values[:-1], df_values)
        ax[0, 2].set_xlabel('T')
        ax[0, 2].set_ylabel(r'$\delta_{f}$')
        ax[0, 2].set_ylim([-15, 15])
        ax[0, 2].grid(True)

        # Plot ax vs time
        ax[1, 2].plot(time_values[:-1],ax_values)
        ax[1, 2].set_xlabel('T(s)')
        ax[1, 2].set_ylabel(r'$a_{x}$')
        ax[1, 2].set_ylim([-5, 5])
        ax[1, 2].grid(True)

        ax[0, 3].plot(x_values,y_values)
        ax[0, 3].set_xlabel('X')
        ax[0, 3].set_ylabel(r'$Y$')
        ax[0, 3].grid(True)


        # ax[1, 3].plot(time_values,caltimeh[:-1])
        # ax[1, 3].set_xlabel('T(s)')
        # ax[1, 3].set_ylabel(r'$cal_time (ms)$')
        # ax[1, 3].grid(True)


        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
