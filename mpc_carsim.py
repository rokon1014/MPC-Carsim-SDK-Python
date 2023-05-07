import time
import math
import casadi
import ctypes
import argparse
import vs_solver
import pickle
import numpy as np
import pandas as pd
from time import sleep
from dataclasses import dataclass
from scipy.spatial import distance
from CarsimRunClass import CarsimRUn
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from vehicle_models import kinematic_model_casadi
from utils import Vehicle, States, get_path_and_lanes

SIMTIME = 50

## MPC FOMULATION WITH CASADI OPTI
TIME_STEP = 0.03
PREDICTION_HORIZON = 8
NUM_OF_STATES = 3
NUM_OF_CONTROLS = 1

## Weight values
WEIGHT_X = 2
WEIGHT_Y = 2
WEIGHT_YAW = 5000
WEIGHT_V = 1
WEIGHT_DELTA = 100
WEIGHT_DIST = 0.445


def get_path_and_lanes():
    """Read excel files and extract path
    and lane points
    """

    # centerline of the road
    df = pd.read_csv("csvs\\converted_gps_normal.csv")
    px = df[df.columns[0]]
    py = df[df.columns[1]]
    pth = df[df.columns[3]]

    # left lane points
    df_left = pd.read_csv("csvs\\lanelet_left.csv")
    px_left = df_left[df_left.columns[0]]
    py_left = df_left[df_left.columns[1]]
    px_left_smoothed = savgol_filter(px_left, 51, 3)

    # right lane points
    df_right = pd.read_csv("csvs\\lanelet_right.csv")
    px_right = df_right[df_right.columns[0]]
    py_right = df_right[df_right.columns[1]]
    px_right_smoothed = savgol_filter(px_right, 51, 3)

    return px, py, pth


def find_closest_point_index(px, py, curx, cury):
    """return of the closest point of the path from the current point"""
    curxy = [(curx, cury)]
    pxy = np.column_stack((px, py))
    ddd = distance.cdist(curxy, pxy, "sqeuclidean")
    min_dist = np.min(ddd)
    min_index = np.argmin(ddd)
    return min_index + 8, min_dist


class MPCTracking:
    """MPC trakcing class"""

    def __init__(self, carsim):
        self.states = [0, 0, 0]  # [x,y, yaw]
        self.goal_pose = [0, 0, 0]
        self.mpc_i = 0
        self.cnt = 0  # delta
        self.cnt_history = []

    def mpc_operation(self):
        tc = 0
        delta = 0  # in radian
        sim_time = 0
        sx = []
        sy = []
        syaw = []
        sdelta = []
        sdist = []

        # get path
        px, py, pth, _ = get_path_and_lanes()

        try:
            # start state
            t, t_step, self.states = csr.check_configuration()
            print(self.states)

            # Run Carsim Simulation for the specified time.
            while sim_time < SIMTIME:
                tc = tc + t_step

                # MPC timer
                # run MPC after each 0.03s. Carsim is running at much higher frequency
                if tc > 0.03:
                    x = self.states[0]
                    y = self.states[1]
                    yaw = math.radians(self.states[2])
                    mpc_current_state = [x, y, yaw]

                    # find the closest reference point from the vehicle
                    closest_index, dist = find_closest_point_index(px, py, x, y)

                    goal = [
                        px[closest_index],
                        py[closest_index],
                        pth[closest_index],
                    ]

                    # Converting goal position to local(vehicle) coordinate
                    # Not used
                    local_x = (goal[0] - x) * math.cos(yaw) + (goal[1] - y) * math.sin(
                        yaw
                    )
                    local_y = -(goal[0] - x) * math.sin(yaw) + (goal[1] - y) * math.cos(
                        yaw
                    )

                    start = time.time()
                    # set initial state of the vehicle from state feedback
                    opti.set_initial(
                        ca_states,
                        np.tile(mpc_current_state, (PREDICTION_HORIZON + 1, 1)),
                    )

                    # set initial value for the control from previous control
                    opti.set_initial(u_dv, np.tile(self.cnt, (PREDICTION_HORIZON, 1)))

                    # filling parameter values
                    opti.set_value(states_current, mpc_current_state)
                    opti.set_value(x_ref, goal[0])
                    opti.set_value(y_ref, goal[1])
                    opti.set_value(yaw_ref, goal[2])

                    # Not used
                    opti.set_value(local_ref_x, goal[0])
                    opti.set_value(local_ref_y, goal[1])

                    try:
                        solution = opti.solve()
                        print("FOUND SOLUTION")
                        c1 = solution.value(u_dv)
                        delta = c1[0]

                    except Exception as e:
                        print("ERROR:", e)
                        greedy_sol = opti.debug.value(u_dv)
                        sol_x = opti.debug.value(x_dv)
                        sol_y = opti.debug.value(y_dv)

                        c1 = opti.debug.value(u_dv)

                        # Using sub-optimal result
                        delta = c1[0]

                    # resetting MPC timer
                    tc = 0

                    # Checking optimization time
                    print("elapsed", time.time() - start)

                    # Saving in history for filtering (smoothing)
                    self.cnt_history.append(delta)

                    # smoothing the steering angle
                    if len(self.cnt_history) > 100:
                        self.cnt_history.pop(0)
                        smoothed = savgol_filter(self.cnt_history, 40, 3)
                        delta = smoothed[-1]

                        print(sim_time, time_step, self.states)
                        sdelta.append(delta)

                        # Keeping in list for saving in CSV
                        sx.append(x)
                        sy.append(y)
                        syaw.append(yaw)
                        sdist.append(local_y)

                # run with a given steering angle at each step
                sim_time, time_step, self.states = csr.run(delta)
                # print(sim_time, time_step, self.states)

            csr.terminate()

            # Save the results in CSV
            result_df = pd.DataFrame({"x": sx, "y": sy, "delta": sdelta, "yaw": syaw})
            date_time = time.strftime("%Y-%m-%d-%H%M%S")
            result_df.to_csv(
                "run_results" + date_time + ".csv",
                index=False,
            )

            # Visualization
            f1 = plt.figure(1)
            plt.plot(sy, sx)

            f2 = plt.figure(2)
            plt.plot(sdelta)

            f3 = plt.figure(3)
            plt.plot(sdist)
            plt.show()

        except Exception as e:
            csr.terminate()
            print(e)
            raise


if __name__ == "__main__":
    sim_file_filename = "gps_path.sim"
    path_to_vs_dll = "carsim_64.dll"

    csr = CarsimRUn(sim_file_filename, path_to_vs_dll)

    vehicle = Vehicle()
    opti = casadi.Opti()

    # State variables [x,y,yaw]
    ca_states = opti.variable(PREDICTION_HORIZON + 1, NUM_OF_STATES)

    # decision variable; bascially states separated
    x_dv = ca_states[:, 0]
    y_dv = ca_states[:, 1]
    yaw_dv = ca_states[:, 2]

    # control variables -> delta
    u_dv = opti.variable(PREDICTION_HORIZON, NUM_OF_CONTROLS)

    ##### PARAMETERS ####
    # references (GOAL!!)
    x_ref = opti.parameter(1)
    y_ref = opti.parameter(1)
    yaw_ref = opti.parameter(1)

    local_ref_x = opti.parameter(1)
    local_ref_y = opti.parameter(1)

    # current state (init)
    states_current = opti.parameter(NUM_OF_STATES)

    ########## CONSTRAINTS and COST###########################
    ### Initial state constraints
    opti.subject_to(x_dv[0] == states_current[0])
    opti.subject_to(y_dv[0] == states_current[1])
    opti.subject_to(yaw_dv[0] == states_current[2])

    cost = 0
    for i in range(PREDICTION_HORIZON):
        st, x_dist, y_dist = kinematic_model_casadi(
            ca_states[i, :], u_dv[i, :], TIME_STEP, vehicle
        )

        opti.subject_to(x_dv[i + 1] == st[0])
        opti.subject_to(y_dv[i + 1] == st[1])
        opti.subject_to(yaw_dv[i + 1] == st[2])

        # cost
        cost_states = (
            WEIGHT_X * ((x_dv[i + 1] - x_ref) - 0.5) ** 2
            + WEIGHT_Y * (y_dv[i + 1] - y_ref) ** 2
            + WEIGHT_YAW * (yaw_dv[i + 1] - yaw_ref) ** 2
        )

        cost = cost + cost_states

    # Constraints on control
    opti.subject_to(u_dv < 0.9)
    opti.subject_to(u_dv > -0.9)

    ## NLP stuff
    ## Using Ipopt for NLP
    opti.minimize(cost)
    s_opts = {
        # "ipopt.max_iter": 10000,
        "ipopt.print_level": 0,
        "ipopt.max_cpu_time": 0.03,
        "print_time": 0,
    }
    # p_opts = {"expand": False}
    # t_opts = {"print_level": 2}
    opti.solver("ipopt", s_opts)

    mpc = MPCTracking(csr)
    mpc.mpc_operation()
