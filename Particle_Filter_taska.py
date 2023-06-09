import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Particle Filter parameters
num_particles = 100  # Number of particles
simulation_time = 200.0 # Simulation time [s]
time_step = 0.05 #ime step [s]

# Motion Model parameters
wheel_radius = 0.1  # [m]

# Noise parameters
motion_noise = np.diag([0.1, 0.15])**2  # Motion noise
measurement_noise = np.diag([0.05, 0.075])**2  # Measurement noise

# State Vector [x y]'
estimated_state = np.zeros((2, 1))
true_state = np.zeros((2, 1))
covariance = np.eye(2)

# Particle store and weights
particles = np.zeros((2, num_particles))  
weights = np.zeros((1, num_particles)) + 1.0 / num_particles  

show_animation = True

def calculate_control_input():
    velocity_right = 1.0  # [m/s]
    velocity_left = 1.0 # [m/s]
    control_input = np.array([[velocity_right, velocity_left]]).T
    return control_input

def simulate_observation(true_state, control_input):
    true_state = calculate_motion(true_state, control_input)

    measurements = []
    z = np.array([[1, 0], [0, 2]]).dot(true_state)
    z[0, 0] += np.random.randn() * measurement_noise[0, 0]
    z[1, 0] += np.random.randn() * measurement_noise[1, 1]
    measurements.append(z)

    measurements = np.array(measurements)

    return true_state, measurements, control_input

def calculate_motion(state, control_input):
    u_r = control_input[0, 0]  # right wheel velocity
    u_l = control_input[1, 0]  # left wheel velocity

    # Update state
    state[0, 0] += wheel_radius / 2 * (u_r + u_l) * time_step + np.random.randn() * motion_noise[0, 0]  # x-position
    state[1, 0] += wheel_radius / 2 * (u_r + u_l) * time_step + np.random.randn() * motion_noise[1, 1]  # y-position

    return state

def calculate_likelihood(x, b):
    p = (1.0 / np.sqrt(2.0 * np.pi * b**2)) * np.exp(-0.5 * (x / b)**2)
    return p

def calculate_covariance(estimated_state, particles, weights):
    cov = np.zeros((2, 2))
    for i in range(particles.shape[1]):
        diff = (particles[:, i].reshape(2,1) - estimated_state)
        cov += weights[0, i] * diff.dot(diff.T)
    return cov

def particle_filter_localization(particles, weights, estimated_state, covariance, z, u):
    for i in range(num_particles):
        x = np.array([particles[:, i]]).T
        weight = 1
        #  Predict with random input sampling
        ud_r = u[0, 0] + np.random.randn() * motion_noise[0, 0]
        ud_l = u[1, 0] + np.random.randn() * motion_noise[1, 1]
        ud = np.array([[ud_r, ud_l]]).T
        x = calculate_motion(x, ud)

        for j in range(len(z)):
            dz = z[j, 0] - x[0, 0]
            da = z[j, 1] - x[1, 0]
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

def main():
    print(__file__ + " start!!")

    current_time = 0.0

    # State Vector [x y]'
    estimated_state = np.zeros((2, 1))
    true_state = np.zeros((2, 1))
    estimated_covariance = np.eye(2)

    particles = np.zeros((2, num_particles))  # Particle store
    particle_weights = np.zeros((1, num_particles)) + 1.0 / num_particles  # Particle weight

    # history
    estimated_state_history = estimated_state
    true_state_history = true_state

    while simulation_time >= current_time:
        current_time += time_step
        control_input = calculate_control_input()

        true_state, z, noisy_control_input = simulate_observation(true_state, control_input)

        estimated_state, estimated_covariance, particles, particle_weights = particle_filter_localization(particles, particle_weights, estimated_state, estimated_covariance, z, noisy_control_input)

        # store data history
        estimated_state_history = np.hstack((estimated_state_history, estimated_state))
        true_state_history = np.hstack((true_state_history, true_state))

        if show_animation:
            plt.cla()
        
            # Plot the true state history as a blue line
            plt.plot(np.array(true_state_history[0, :]).flatten(),
                     np.array(true_state_history[1, :]).flatten(), "-b")
            # Plot the estimated state history as a red line
            plt.plot(np.array(estimated_state_history[0, :]).flatten(),
                     np.array(estimated_state_history[1, :]).flatten(), "-r")
            # Plot the particles as green dots
            plt.scatter(particles[0, :], particles[1, :], color='g')
            
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()

