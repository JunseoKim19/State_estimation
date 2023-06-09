import numpy as np
import scipy.linalg
import math
import matplotlib.pyplot as plt

# Particle Filter parameters
num_particles = 100  # Number of particles
simulation_time = 200.0 # Simulation time [s]
time_step = 0.05 #ime step [s]

# Motion Model parameters
wheel_radius = 0.1  # [m]
wheelbase = 0.3  # [m]

# Single Landmark
landmarks = np.array([[10.0, 10.0]])

# Noise parameters
motion_noise = np.diag([0.1, np.deg2rad(0.01)])**2  # Motion noise
measurement_noise = np.diag([0.01, 0.1])**2  # Measurement noise

# State Vector [x y yaw v]'
estimated_state = np.zeros((4, 1))
true_state = np.zeros((4, 1))
covariance = np.eye(4)

# Particle store and weights
particles = np.zeros((4, num_particles))  
weights = np.zeros((1, num_particles)) + 1.0 / num_particles  

show_animation = True

def calculate_control_input():
    velocity = 10.0  # [m/s]
    yaw_rate = 0.2  # [rad/s]
    control_input = np.array([[velocity, yaw_rate]]).T
    return control_input

def simulate_observation(true_state, dead_reckoning_state, control_input):
    true_state = calculate_motion(true_state, control_input)

    measurements = []
    for landmark in landmarks:
        dx = true_state[0, 0] - landmark[0]
        dy = true_state[1, 0] - landmark[1]
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx) - true_state[2, 0]
        
        # add noise to the measurements
        noisy_distance = distance + np.random.randn() * measurement_noise[0, 0]
        noisy_angle = angle + np.random.randn() * measurement_noise[1, 1]
        
        measurements.append([noisy_distance, noisy_angle])

    measurements = np.array(measurements)

    # use the true control input for the dead reckoning state
    dead_reckoning_state = calculate_motion(dead_reckoning_state, control_input)

    return true_state, measurements, dead_reckoning_state, control_input


def calculate_motion(state, control_input):
    u_omega = control_input[0, 0]  # velocity
    u_psi = control_input[1, 0]  # steering angle

    # Update state
    state[0, 0] += wheel_radius * u_omega * np.cos(state[2, 0]) * time_step  # x-position
    state[1, 0] += wheel_radius * u_omega * np.sin(state[2, 0]) * time_step  # y-position
    state[2, 0] += wheel_radius / wheelbase * u_psi * time_step  # yaw angle
    state[3, 0] = u_omega  # velocity

    return state


def calculate_likelihood(x, b):
    p = (1.0 / (2.0 * b)) * np.exp(-np.abs(x) / b)
    return p

def calculate_covariance(estimated_state, particles, weights):
    cov = np.zeros((4, 4))
    for i in range(particles.shape[1]):
        diff = (particles[:, i].reshape(4,1) - estimated_state)
        cov += weights[0, i] * diff.dot(diff.T)
    return cov

def particle_filter_localization(particles, weights, estimated_state, covariance, z, u):
    for i in range(num_particles):
        x = np.array([particles[:, i]]).T
        weight = 1
        #  Predict with random input sampling
        ud1 = u[0, 0] + np.random.randn() * motion_noise[0, 0]
        ud2 = u[1, 0] + np.random.randn() * motion_noise[1, 1]
        ud = np.array([[ud1, ud2]]).T
        x = calculate_motion(x, ud)

        for j in range(len(z[:, 0])):
            dx = x[0, 0] - landmarks[j, 0]
            dy = x[1, 0] - landmarks[j, 1]
            prez = math.sqrt(dx**2 + dy**2)
            prea = math.atan2(dy, dx) - x[2, 0]
            dz = prez - z[j, 0]
            da = prea - z[j, 1]
            weight *= calculate_likelihood(dz, np.sqrt(measurement_noise[0, 0]))
            weight *= calculate_likelihood(da, np.sqrt(measurement_noise[1, 1]))

        particles[:, i] = x[:, 0]
        weights[0, i] = weight

    weights = weights / (weights.sum() + 1e-10)  # normalize

    estimated_state = particles.dot(weights.T)
    covariance = calculate_covariance(estimated_state, particles, weights)

    covariance_sqrt = scipy.linalg.sqrtm(covariance)
    for i in range(num_particles):
        particles[:,i] = estimated_state.reshape(-1,) + np.diag(covariance_sqrt) * \
            np.random.randn(estimated_state.shape[0])

    return estimated_state, covariance, particles, weights


def plot_error_ellipse(estimated_state, estimated_covariance):  
    covariance_xy = estimated_covariance[0:2, 0:2]
    eigenvalues, eigenvectors = np.linalg.eig(covariance_xy)

    if eigenvalues[0] >= eigenvalues[1]:
        large_index = 0
        small_index = 1
    else:
        large_index = 1
        small_index = 0

    angle_range = np.arange(0, 2 * math.pi + 0.1, 0.1)

    try:
        large_axis = math.sqrt(eigenvalues[large_index])
    except ValueError:
        large_axis = 0

    try:
        small_axis = math.sqrt(eigenvalues[small_index])
    except ValueError:
        small_axis = 0

    ellipse_x = [large_axis * math.cos(theta) for theta in angle_range]
    ellipse_y = [small_axis * math.sin(theta) for theta in angle_range]
    rotation_angle = math.atan2(eigenvectors[large_index, 1], eigenvectors[large_index, 0])
    rotation_matrix = np.array([[math.cos(rotation_angle), math.sin(rotation_angle)],
                                [-math.sin(rotation_angle), math.cos(rotation_angle)]])
    rotated_ellipse = rotation_matrix.dot(np.array([[ellipse_x, ellipse_y]]))
    plot_x = np.array(rotated_ellipse[0, :] + estimated_state[0, 0]).flatten()
    plot_y = np.array(rotated_ellipse[1, :] + estimated_state[1, 0]).flatten()
    plt.plot(plot_x, plot_y, "--r")


def main():
    print(__file__ + " start!!")

    current_time = 0.0

    # State Vector [x y yaw v]'
    estimated_state = np.zeros((4, 1))
    true_state = np.zeros((4, 1))
    estimated_covariance = np.eye(4)

    particles = np.zeros((4, num_particles))  # Particle store
    particle_weights = np.zeros((1, num_particles)) + 1.0 / num_particles  # Particle weight
    dead_reckoning = np.zeros((4, 1))  # Dead reckoning

    # history
    estimated_state_history = estimated_state
    true_state_history = true_state
    dead_reckoning_history = true_state

    while simulation_time >= current_time:
        current_time += time_step
        control_input = calculate_control_input()

        true_state, z, dead_reckoning, noisy_control_input = simulate_observation(true_state, dead_reckoning, control_input)

        estimated_state, estimated_covariance, particles, particle_weights = particle_filter_localization(particles, particle_weights, estimated_state, estimated_covariance, z, noisy_control_input)

        # store data history
        estimated_state_history = np.hstack((estimated_state_history, estimated_state))
        dead_reckoning_history = np.hstack((dead_reckoning_history, dead_reckoning))
        true_state_history = np.hstack((true_state_history, true_state))

        if show_animation:
            plt.cla()
        
            for i in range(len(z)):
                for j in range(len(landmarks)):
                    plt.plot([true_state[0, 0], landmarks[j, 0]], [true_state[1, 0], landmarks[j, 1]], "-k")
            for i in range(len(landmarks)):
                plt.plot(landmarks[i, 0], landmarks[i, 1], "*k")
        
            # Normalize the weights to be between 0 and 1
            normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
        
            # Create a colormap
            cmap = plt.cm.get_cmap("jet")
        
            # Plot the particles with color and size based on weight
            # Plot the particles with color and size based on weight
            plt.scatter(particles[0, :], particles[1, :], c=cmap(weights.flatten()), s=weights.flatten() * 1000)

        
            plt.plot(np.array(true_state_history[0, :]).flatten(),
                     np.array(true_state_history[1, :]).flatten(), "-b")
            plt.plot(np.array(dead_reckoning_history[0, :]).flatten(),
                     np.array(dead_reckoning_history[1, :]).flatten(), "-k")
            plt.plot(np.array(estimated_state_history[0, :]).flatten(),
                     np.array(estimated_state_history[1, :]).flatten(), "-r")
            plot_error_ellipse(estimated_state, estimated_covariance)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()