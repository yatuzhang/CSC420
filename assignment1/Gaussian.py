import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

mu = 0
variance = 4
sigma = math.sqrt(variance)
x = np.linspace(-6,6, 100)

plt.plot(x,mlab.normpdf(x, mu, sigma))
plt.title("One Dimensional Gaussian with Sigma=2")
plt.ylabel('Gaussian Value')
plt.xlabel('Y (position from origin 0)')
plt.axis([-6, 6, 0, 0.25])
plt.show()