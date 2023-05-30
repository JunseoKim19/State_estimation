import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Degrees of freedom
K_values = [1, 2, 3, 10, 100]

x = np.linspace(0, max(K_values), 1000)

for K in K_values:
    y = chi2.pdf(x, K)
    plt.plot(x, y, label=f'K = {K}')

plt.title('y=x^T*T')
plt.xlabel('x')
plt.ylabel('Probability Density Function')
plt.legend()
plt.grid(True)
plt.show()

