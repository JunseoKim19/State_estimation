
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

N = 1000


mu = np.array([1, np.deg2rad(0.5)])  
Sigma = np.array([[0.01, 0], [0, 0.005]]) 


def polar_to_cartesian(rho, theta):
    x = rho * np.cos(theta)  
    y = rho * np.sin(theta)  
    return np.array([x, y])  


x_samples = np.random.multivariate_normal(mu, Sigma, N)


y_samples = []


for sample in x_samples:
    rho, theta = sample 
    y = polar_to_cartesian(rho, theta) 
    y_samples.append(y) 


y_samples = np.array(y_samples)


mu_y = np.mean(y_samples, axis=0)
Sigma_y = np.cov(y_samples.T)


plt.scatter(y_samples[:, 0], y_samples[:, 1], s=5)


lambda_, v = np.linalg.eig(Sigma_y)
lambda_ = np.sqrt(lambda_)
angle = np.rad2deg(np.arccos(v[0, 0]))


ell = Ellipse(xy=(mu_y[0], mu_y[1]),
              width=lambda_[0]*2, height=lambda_[1]*2,
              angle=angle,
              edgecolor='r', fc='None', lw=2)


plt.gca().add_patch(ell)

plt.title('Transformed results and uncertainty ellipse')
plt.xlabel('x')
plt.ylabel('y')

plt.grid(True)

plt.axis('equal')

plt.show()
