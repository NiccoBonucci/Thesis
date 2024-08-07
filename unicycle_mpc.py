from casadi import *
import casadi as cs

from rockit import *

from pylab import *
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
add_noise = True            # enable/disable the measurement noise addition in simulation
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
ocp = Ocp(T=FreeTime(Tf))

# Define states
x = ocp.state()  # [m]
y = ocp.state()  # [m]
theta  = ocp.state()  # [rad]
X = vertcat(x,y,theta)

# Define controls

V = ocp.control()
omega = ocp.control()

F = vertcat(V, omega)

# Specify ODE
ocp.set_der(x, V * cs.cos(theta))
ocp.set_der(y, V * cs.sin(theta))
ocp.set_der(theta, omega)

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==0)
ocp.subject_to(ocp.at_t0(y)==0)
ocp.subject_to(ocp.at_t0(theta)==0)

# Final constraint
ocp.subject_to(ocp.at_tf(x)==3)
ocp.subject_to(ocp.at_tf(y)==4)
ocp.subject_to(ocp.at_tf(theta)==3.14)

# Path constraints
ocp.set_initial(x,0)
ocp.set_initial(y,ocp.t)
ocp.set_initial(theta, 0)
ocp.set_initial(V,1)
ocp.set_initial(omega,0)

ocp.subject_to(0 <= (V<=2))
ocp.subject_to( -pi <= (omega<= pi))
ocp.subject_to(-10 <= (x <= 10), include_first=False)
ocp.subject_to(-10 <= (y <= 10), include_first=False)

# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt',options)

method = MultipleShooting(N=Nhor, M=4, intg='rk')
ocp.method(method)

# Minimal time
ocp.add_objective(ocp.T)
ocp.add_objective(ocp.integral((x)**2 + (y)**2))

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
    # Simulate dynamics and update the current state
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

    # Solve the optimization problem
    sol = ocp.solve()

    x_history[i+1] = current_X[0].full()
    y_history[i+1] = current_X[1].full()
    theta_history[i+1] = current_X[2].full()
    u_history[i-1,:] = F_sol_first

print("Plot the results")

"""
time_sim = np.linspace(0, dt*Nsim, Nsim+1)

figure()

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')

plot(xs, ys,'bo')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')

plot(xs, ys, 'b.')

ts, xs = sol.sample(x, grid='integrator',refine=10)
ts, ys = sol.sample(y, grid='integrator',refine=10)

plot(xs, ys, '-')

axis('equal')
show(block=True)
"""

# Plotting the trajectory
plt.figure()
plt.plot(x_history, y_history, marker='o')
plt.title('Traiettoria del robot uniciclo')
plt.xlabel('Posizione X [m]')
plt.ylabel('Posizione Y [m]')
plt.grid(True)
plt.xlim(-10, 10)  # limiti dell'asse X (aggiustabili in base ai tuoi vincoli)
plt.ylim(-10, 10)  # limiti dell'asse Y (aggiustabili in base ai tuoi vincoli)
plt.show()