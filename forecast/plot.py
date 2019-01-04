import numpy as np
import matplotlib.pyplot as plt

data = np.load('global_results.npy')

plt.figure()
plt.plot(data[:,0], label='Numerical model')
plt.plot(data[:,1], label='Neural net')
plt.legend()

plt.show()