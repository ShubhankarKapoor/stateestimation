import numpy as np
import matplotlib.pyplot as plt
import time

from networks import Network
from solvers import backwardforwardsweep, lindistflowsweep

""" 
    Demonstrates code for backwardforwardsweep solver
    Assumes LV network, PQ loads only, and slack bus is the first bus (i.e. bus 0)
"""


network37 = Network('network37', sparse=False)  # the sparse option might give speed improvements on very large networks

# meas from full AC
ts = time.time()
V_all, line_currents, V_mag, V_ang, S_line, max_diff, diff_save = backwardforwardsweep(network37)
tf = time.time()

print('Time to solve = ', tf-ts, 's')

plt.semilogy(np.array(diff_save),'-sk')
plt.xlabel('Iteration')
plt.ylabel('Max value change')
plt.title("backward forward sweep solver convergence")
plt.show()

network37 = Network('network13', sparse=False)  # the sparse option might give speed improvements on very large networks

# meas from lindistflow
ts = time.time()
V_all, V_mag, P_line, Q_line, S_line, max_diff, diff_save, k = lindistflowsweep(network37, pflow = 1)
tf = time.time()

print('Time to solve = ', tf-ts, 's')
