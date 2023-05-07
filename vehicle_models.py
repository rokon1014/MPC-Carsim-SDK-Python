import math
import casadi


def kinematic_model(states, controls, time_step, vehicle_params):
    lf = vehicle_params.lf
    lr = vehicle_params.lr
    L = lf + lr

    x = states[0]
    y = states[1]
    yaw = states[2]
    vx = 5

    # wheel angle
    # TODO: add options for multiple control
    delta = controls

    beta = math.atan2(lr * math.tan(delta), L)

    dx = vx * math.cos(yaw + delta)
    dy = vx * math.sin(yaw + delta)
    dyaw = (vx * math.cos(beta) * math.tan(delta)) / L

    x = x + (dx * time_step)
    y = y + (dy * time_step)
    yaw = yaw + (dyaw * time_step)

    states[0] = x
    states[1] = y
    states[2] = yaw

    return states


def kinematic_model_casadi(states, control, TIME_STEP, vehicle):
    """vehicle kinematic model"""
    _x = states[0]
    _y = states[1]
    _yaw = states[2]

    _delta = control[0]
    _vx = 5

    _beta = casadi.atan2(vehicle.lr * casadi.tan(_delta), (vehicle.lf + vehicle.lr))

    _dx = _vx * casadi.cos(_yaw + _beta)
    _dy = _vx * casadi.sin(_yaw + _beta)
    _dyaw = (_vx * casadi.cos(_beta) * casadi.tan(_delta)) / (vehicle.lf + vehicle.lr)

    _x = _x + _dx * TIME_STEP
    _y = _y + _dy * TIME_STEP
    _yaw = _yaw + _dyaw * TIME_STEP
    _st = [_x, _y, _yaw]

    return _st


def kinematic_model_casadi_local(states, control, TIME_STEP, vehicle, goal_x, goal_y):
    """vehicle kinematic model"""
    _x = states[0]
    _y = states[1]
    _yaw = states[2]

    _delta = control[0]
    _vx = 5

    _beta = casadi.atan2(vehicle.lr * casadi.tan(_delta), (vehicle.lf + vehicle.lr))

    _dx = _vx * casadi.cos(_yaw + _beta)
    _dy = _vx * casadi.sin(_yaw + _beta)
    _dyaw = (_vx * casadi.cos(_beta) * casadi.tan(_delta)) / (vehicle.lf + vehicle.lr)

    _x = _x + _dx * TIME_STEP
    _y = _y + _dy * TIME_STEP
    _yaw = _yaw + _dyaw * TIME_STEP
    _st = [_x, _y, _yaw]

    local_x = (goal_x - _x) * casadi.cos(_yaw) + (goal_y - _y) * casadi.sin(_yaw)
    local_y = -(goal_x - _x) * casadi.sin(_yaw) + (goal_y - _y) * casadi.cos(_yaw)

    return _st, local_x, local_y
