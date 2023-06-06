import pygame
import numpy as np

r = 0.1
ur = ul = 0.1
T = 1/8
x0 = np.array([0, 0])
P0 = np.array([[0, 0], [0, 0]])
Q = np.array([[0.1, 0], [0, 0.15]])
R = np.array([[0.05, 0], [0, 0.075]])

x = x0
P = P0
prev_x = x0

pygame.init()
screen = pygame.display.set_mode((600, 450))
clock = pygame.time.Clock()

scale_factor = 10

trajectory = [x0]
predicted_trajectory = [x0]

def prediction_step(x, prev_x, P, Q):
    noise = np.random.normal(0, 0.01, size=2)  # Reduced noise
    x = prev_x + r/2 * (ur + ul) * np.array([1, 1]) * T + noise
    P = P + Q
    return x, P

def ekf_correction_step(x, P, R):
    # Jacobian of H
    H_j = np.eye(2)
    
    # Extended Kalman Filter Gain
    K = np.dot(np.dot(P, H_j.T), np.linalg.inv(np.dot(np.dot(H_j, P), H_j.T) + R))
    
    # Update step
    x = x + np.dot(K, (np.dot(H_j, x) + np.random.normal(0, np.sqrt(R[0, 0])/2)))
    P = np.dot((np.eye(2) - np.dot(K, H_j)), P)
    
    return x, P

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    x, P = prediction_step(x, prev_x, P, Q)
    predicted_trajectory.append(x)
    if pygame.time.get_ticks() % 1000 == 0:
        x, P = ekf_correction_step(x, P, R)
        trajectory.append(x)

    prev_x = x
    
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (0, 255, 0), (int(x[0]*scale_factor), int(x[1]*scale_factor)), 5)

    U, s, _ = np.linalg.svd(P)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 1 * np.sqrt(s)
    ellipse_surface = pygame.Surface((width*scale_factor, height*scale_factor), pygame.SRCALPHA)
    pygame.draw.ellipse(ellipse_surface, (0, 0, 255, 100), pygame.Rect(0, 0, width*scale_factor, height*scale_factor))
    rotated_ellipse = pygame.transform.rotate(ellipse_surface, -angle)
    screen.blit(rotated_ellipse, (x[0]*scale_factor - rotated_ellipse.get_width()/2, x[1]*scale_factor - rotated_ellipse.get_height()/2))
    
    if len(trajectory) > 1:
        pygame.draw.lines(screen, (255, 255, 255), False, [(x[0]*scale_factor, x[1]*scale_factor) for x in trajectory], 3)
    if len(predicted_trajectory) > 1:
        pygame.draw.lines(screen, (255, 0, 0), False, [(x[0]*scale_factor, x[1]*scale_factor) for x in predicted_trajectory], 3)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
