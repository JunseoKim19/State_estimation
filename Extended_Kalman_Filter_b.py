import sys
import pathlib
import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation as Rot

def rotation_matrix_2d(angle):
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def angle_mod(x, zero_2_2pi=False, degree=False):
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)
    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
    
# Covariance for EKF simulation
state_covariance = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
measurement_covariance = np.diag([0.1, np.deg2rad(1.0)]) ** 2  

#  Simulation parameter
RANGE_NOISE = 0.1
BEARING_NOISE = 0.01

DT = 0.02 

show_animation = True

# Landmark position
landmark_position = np.array([10, 10])

    
def calculate_input():
    velocity = 1.0 
    yaw_rate = 1.0 
    control_input = np.array([[velocity], [yaw_rate]])
    return control_input


def observation(true_state, control_input):
    true_state = motion_model(true_state, control_input)

    measurement = observation_model(true_state) + np.array([[np.random.normal(0, RANGE_NOISE)], [np.random.normal(0, BEARING_NOISE)]])

    return true_state, measurement, control_input

def motion_model(state, control_input):
    r = 0.1  # radius of the wheel
    L = 0.3  # distance between the wheels
    u_r = control_input[0, 0]  # control signal to the right wheel
    u_l = control_input[1, 0]  # control signal to the left wheel
    u_w = 0.5 * (u_r + u_l)  # average of the control signals
    u_psi = u_r - u_l  # difference of the control signals

    # Motion model equations
    state[0, 0] += DT * r * u_w * math.cos(state[2, 0])
    state[1, 0] += DT * r * u_w * math.sin(state[2, 0])
    state[2, 0] += DT * r * u_psi / L
    state[2, 0] = angle_mod(state[2, 0])

    return state

def observation_model(state):
    dx = landmark_position[0] - state[0, 0]
    dy = landmark_position[1] - state[1, 0]
    distance = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx) - state[2, 0]
    measurement = np.array([[distance], [angle]])
    return measurement

def jacobian_of_motion_model(state, control_input):
    yaw = state[2, 0]
    velocity = control_input[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * velocity * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * velocity * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF

def jacobian_of_observation_model(state):
    dx = landmark_position[0] - state[0, 0]
    dy = landmark_position[1] - state[1, 0]
    distance = math.sqrt(dx**2 + dy**2)
    H = np.array([
        [-dx / distance, -dy / distance, 0, 0],
        [dy / (distance**2), -dx / (distance**2), -1, 0]
    ])
    return H

def ekf_estimation(state_estimate, covariance_estimate, measurement, control_input):
    #  Prediction
    state_predict = motion_model(state_estimate, control_input)
    jF = jacobian_of_motion_model(state_estimate, control_input)
    covariance_predict = jF @ covariance_estimate @ jF.T + state_covariance

    #  Update
    jH = jacobian_of_observation_model(state_estimate)
    predicted_measurement = observation_model(state_predict)
    innovation = measurement - predicted_measurement
    innovation[1] = angle_mod(innovation[1])
    innovation_covariance = jH @ covariance_predict @ jH.T + measurement_covariance
    kalman_gain = covariance_predict @ jH.T @ np.linalg.inv(innovation_covariance)
    state_estimate = state_predict + kalman_gain @ innovation
    covariance_estimate = (np.eye(len(state_estimate)) - kalman_gain @ jH) @ covariance_predict
    return state_estimate, covariance_estimate

def plot_covariance_ellipse(state_estimate, covariance_estimate):
    Pxy = covariance_estimate[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    fx = rotation_matrix_2d(angle) @ (np.array([x, y]))
    px = np.array(fx[0, :] + state_estimate[0, 0]).flatten()
    py = np.array(fx[1, :] + state_estimate[1, 0]).flatten()
    plt.plot(px, py, "--r")

def main():
    print(__file__ + " start!!")

    # State Vector [x y yaw v]'
    state_estimate = np.zeros((4, 1))
    true_state = np.zeros((4, 1))
    covariance_estimate = np.eye(4)

    history_state_estimate = state_estimate
    history_true_state = true_state
    history_measurement = np.zeros((2, 1))

    plt.figure()  

    while True: 
        control_input = calculate_input()

        true_state, measurement, control_input = observation(true_state, control_input)

        state_estimate, covariance_estimate = ekf_estimation(state_estimate, covariance_estimate, measurement, control_input)

        history_state_estimate = np.hstack((history_state_estimate, state_estimate))
        history_true_state = np.hstack((history_true_state, true_state))
        history_measurement = np.hstack((history_measurement, measurement))

        if show_animation:
            plt.cla()  
            plt.plot(history_state_estimate[0, :].flatten(),
                     history_state_estimate[1, :].flatten(), "-r")
            plt.plot(landmark_position[0], landmark_position[1], "go") 
            plot_covariance_ellipse(state_estimate, covariance_estimate)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.000001) 

if __name__ == '__main__':
    main()


