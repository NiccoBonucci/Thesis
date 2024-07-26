
from casadi import *
import casadi as cs

from rockit import *

import numpy as np
import matplotlib.pyplot as plt

from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time


nx    = 3                   # the system is composed of 4 states
nu    = 2                   # the system has 1 input
Tf    = 5.0                 # control horizon [s]
Nhor  = 50                  # number of control intervals
dt    = Tf/Nhor             # sample time

current_X = vertcat(0,0,0)  # initial state
final_X   = vertcat(3,4,0)    # desired terminal state

Nsim  = 100                  # how much samples to simulate
add_noise = False            # enable/disable the measurement noise addition in simulation
add_disturbance = False      # enable/disable the disturbance addition in simulation

# -------------------------------
# Logging variables
# -------------------------------
x_history = np.zeros(Nsim+1)
y_history = np.zeros(Nsim+1)
theta_history = np.zeros(Nsim+1)
u_history = np.zeros((Nsim, 2))
# -------------------------------
# Set OCP
# -------------------------------
ocp = Ocp(T=Tf)

# Define states
x = ocp.state()  # [m]
y = ocp.state()  # [m]
theta  = ocp.state()  # [rad]
X = vertcat(x,y,theta)

# Define controls

v = ocp.control()
omega = ocp.control()

F = vertcat(v, omega)

# Define parameter
X_0 = ocp.parameter(nx)

# Specify ODE
ocp.set_der(x, v * cs.cos(theta))
ocp.set_der(y, v * cs.sin(theta))
ocp.set_der(theta, omega)

# Initial and final constraints
ocp.subject_to(ocp.at_t0(X)==X_0)
ocp.subject_to(ocp.at_tf(X)==final_X)

# Path constraints
ocp.subject_to(-2 <= (F <= 2 ))
ocp.subject_to(-5 <= (x <= 5), include_first=False)
ocp.subject_to(-5 <= (y <= 5), include_first=False)

# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt',options)

method = MultipleShooting(N=Nhor, intg='rk')
ocp.method(method)


# Set initial value for parameters
ocp.set_value(X_0, current_X)

# Solve
sol = ocp.solve()

Sim_unycicle_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
x_history[0]   = current_X[0]
x_history[0]   = current_X[1]
theta_history[0] = current_X[2]

DM.rng(0)

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    tsa, Fsol = sol.sample(F, grid='control')
    F_sol_first = Fsol[0, :]
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_unycicle_dyn(x0=current_X, u=F_sol_first, T=dt)["xf"]
    # Add disturbance at t = 2*Tf
    if add_disturbance:
        if i == round(2*Nhor)-1:
            disturbance = vertcat(0,0,-1e-1)
            current_X = current_X + disturbance
    # Add measurement noise
    if add_noise:
        meas_noise = 5e-4*(DM.rand(nx,1)-vertcat(1,1,1)) # 3x1 vector with values in [-1e-3, 1e-3]
        current_X = current_X + meas_noise
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X[:3])
    # Solve the optimization problem
    sol = ocp.solve()

    x_history[i+1] = current_X[0].full()
    y_history[i+1] = current_X[1].full()
    theta_history[i+1] = current_X[2].full()
    u_history[i-1,:] = F_sol_first

print("Plot the results")

time_sim = np.linspace(0, dt*Nsim, Nsim+1)

"""
fig, ax1 = plt.subplots()
ax1.plot(time_sim, x_history, 'r-', label='x position')
ax1.plot(time_sim, y_history, 'b-', label='y position')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Position [m]')
ax1.legend()
plt.show()

fig, ax2 = plt.subplots()
ax2.plot(time_sim[:-1], theta_history[:-1], 'g-', label='theta')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Theta [rad]')
ax2.legend()
plt.show()
"""

# Plot the trajectory (x vs y)
fig, ax3 = plt.subplots()
ax3.plot(x_history, y_history, 'b-', label='Trajectory')
ax3.plot(x_history[0], y_history[0], 'go', label='Start')
ax3.plot(x_history[-1], y_history[-1], 'ro', label='End')
ax3.set_xlabel('x position [m]')
ax3.set_ylabel('y position [m]')
ax3.legend()
ax3.grid()
plt.title("Trajectory of the unicycle")
plt.show()