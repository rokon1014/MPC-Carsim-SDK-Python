from CarsimRunClass import CarsimRUn
import argparse
import ctypes
from time import sleep
import vs_solver
import math
from vehicle_models import (
    kinematic_model,
    kinematic_model_casadi,
    kinematic_model_casadi_local,
)
from dataclasses import dataclass
from matplotlib import pyplot as plt
from utils import Vehicle, States, get_path_and_lanes
import casadi
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.signal import savgol_filter
import pickle
import time

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
WEIGHT_DELTA =100
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

    # plt.plot(py_left, px_left_smoothed, "b")
    # plt.plot(py_left, px_left, "r")

    return px, py, px_left_smoothed, py_left, px_right_smoothed, py_right


def find_closest_point_index(px, py, curx, cury):
    """return of the closest point of the path from the current point"""
    curxy = [(curx, cury)]
    pxy = np.column_stack((px, py))
    ddd = distance.cdist(curxy, pxy, "sqeuclidean")
    min_dist = np.min(ddd)
    min_index = np.argmin(ddd)
    return min_index + 8 , min_dist


class MPCTracking:
    """MPC trakcing class"""

    def __init__(self, carsim):
        self.states = [0, 0, 0]  # [x,y, yaw]
        self.goal_pose = [0, 0, 0]
        self.mpc_i = 0
        self.cnt = 0  # delta
        self.cnt_history= []

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
        _, _, px_left, py_left, px_right, py_right = get_path_and_lanes()

        try:
            # start state
            t, t_step, self.states = csr.check_configuration()
            print(self.states)

            while sim_time < SIMTIME:
                tc = tc + t_step

                if tc > 0.03:
                    x = self.states[0]
                    y = self.states[1]
                    yaw = math.radians(self.states[2])
                    mpc_current_state = [x, y, yaw]

                    closest_index, dist = find_closest_point_index(px_right, py_right, x, y)

                    goal_left = [px_left[closest_index], py_left[closest_index]]

                    local_x = (goal_left[0] - x) * math.cos(yaw) + (goal_left[1] - y) * math.sin(
                        yaw
                    )
                    local_y = -(goal_left[0] - x) * math.sin(yaw) + (goal_left[1] - y) * math.cos(
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
                    opti.set_value(x_ref, goal_left[0])
                    opti.set_value(y_ref, goal_left[1])

                    opti.set_value(local_ref_x, goal_left[0])
                    opti.set_value(local_ref_y, goal_left[1])

                    try:
                        solution = opti.solve()
                        # print("FOUND SOLUTION")
                        c1 = solution.value(u_dv)
                        delta = c1[0]

                    except Exception as e:
                        print("ERROR:", e)
                        greedy_sol = opti.debug.value(u_dv)
                        sol_x = opti.debug.value(x_dv)
                        sol_y = opti.debug.value(y_dv)

                        c1 = opti.debug.value(u_dv)
                        delta = c1[0]

                    # plt.plot(y, x, "*")
                    # plt.plot(goal_left[1], goal_left[0], "o")
                    # plt.draw()
                    # plt.pause(0.001)

                    # resetting tc
                    tc = 0

                    print("elapsed", time.time()-start)
                   
                    self.cnt_history.append(delta)
                    # print("HIST", self.cnt_history)


                    # smoothing the steering angle
                    if len(self.cnt_history)>100:
                        self.cnt_history.pop(0)
                        smoothed = savgol_filter(self.cnt_history, 40, 3)
                        delta = smoothed[-1]

                        print(sim_time, time_step, self.states)
                        sdelta.append(delta)
                        
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
                "run_results\\result-right-lane" + ".csv",
                index=False,
            )


            # Visualization
            f1= plt.figure(1)
            plt.plot(sy, sx)
            plt.plot(py_left,px_left)
            plt.plot(py_right, px_right)

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
        st, x_dist, y_dist = kinematic_model_casadi_local(
            ca_states[i, :], u_dv[i, :], TIME_STEP, vehicle, local_ref_x, local_ref_y
        )

        opti.subject_to(x_dv[i + 1] == st[0])
        opti.subject_to(y_dv[i + 1] == st[1])
        opti.subject_to(yaw_dv[i + 1] == st[2])

        # cost
        # cost_states = (
        #     WEIGHT_X * ((x_dv[i + 1] - x_ref) - 0.5) ** 2
        #     + WEIGHT_Y * (y_dv[i + 1] - y_ref) ** 2
        #     + WEIGHT_YAW * (yaw_dv[i + 1] - yaw_ref) ** 2
        # )

        # cost = cost + cost_states

        cost_states = WEIGHT_X * (x_dist) ** 2 + WEIGHT_Y * (y_dist - WEIGHT_DIST) ** 2 +  WEIGHT_YAW * (yaw_dv[i + 1] - yaw_dv[i ]) ** 2
        cost = cost + cost_states

        if i < PREDICTION_HORIZON - 1:
            cost_u = WEIGHT_DELTA * (u_dv[i + 1] - u_dv[i]) ** 2
            cost = cost + cost_u

    opti.subject_to(u_dv < 0.9)
    opti.subject_to(u_dv > -0.9)

    ## NLP stuff
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
