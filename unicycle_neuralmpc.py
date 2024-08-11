import torch
import torch.nn as nn
import l4casadi as l4c

import casadi as cs
from casadi import *

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
final_F   = vertcat(0,0)    # desired terminal state

Nsim  = 45                  # how much samples to simulate
add_noise = False            # enable/disable the measurement noise addition in simulation
add_disturbance = False      # enable/disable the disturbance addition in simulation

############################################################################
############# Definizione della dinamica tramite rete neurale ##############
############################################################################

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(5, 512)  # 5 input nodes: x, y, theta, v, omega
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)  # 3 output nodes: x_next, y_next, theta_next
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the dynamic system using the trained model and l4casADi
PyTorch_model = PyTorchModel()
PyTorch_model.load_state_dict(torch.load("unicycle_model_state.pth"))
PyTorch_model.eval()
learned_dyn = l4c.L4CasADi(PyTorch_model, model_expects_batch_dim=True, device='cpu')

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
X = cs.vertcat(x,y,theta)

# Define controls
V = ocp.control()
omega = ocp.control()
F = cs.vertcat(V, omega)

in_sym = cs.vertcat(X, F)

# Specify ODE
ocp.set_der(x, V * cs.cos(theta))
ocp.set_der(y, V * cs.sin(theta))
ocp.set_der(theta, omega)

# Define parameter
X_0 = ocp.parameter(nx)

# Initial constraint
ocp.subject_to(ocp.at_t0(X)==X_0)

# Final constraint on state
ocp.subject_to(ocp.at_tf(X)==final_X)

# Final constraint on control
ocp.subject_to(ocp.at_tf(F)==final_F)

# Path constraints
ocp.set_initial(x,0)
ocp.set_initial(y,0)
ocp.set_initial(theta, 0)
ocp.set_initial(V,1)
ocp.set_initial(omega,0)

ocp.subject_to(0 <= (V<=2))
ocp.subject_to( -2 <= (omega<= 2))
ocp.subject_to(-5 <= (x <= 5), include_first=False)
ocp.subject_to(-5 <= (y <= 5), include_first=False)
ocp.subject_to( -pi <= (theta<= pi), include_first=False)

# Pick a solution method
ocp.solver('ipopt')

method = MultipleShooting(N=Nhor, M=4, intg='rk')
ocp.method(method)

# Minimal time
ocp.add_objective(ocp.T)
ocp.add_objective(ocp.integral((x)**2 + (y)**2 + (theta)**2))

# Set initial value for parameters
ocp.set_value(X_0, current_X)

# Solve
sol = ocp.solve()

###################################################
############ Grafico soluzione iniziale ###########
###################################################

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

# Log data for post-processing
x_history[0]   = current_X[0]
y_history[0]   = current_X[1]
theta_history[0] = current_X[2]

DM.rng(0)

for i in range(Nsim):
    print("   ")
    print("timestep", i+1, "of", Nsim)
    print("   ")

    # Get the solution from sol
    tsa, Fsol = sol.sample(F, grid='control')
    F_sol_first = Fsol[0, :]
    #print(type(current_X))
    # Convert current_X to DM for compatibility with CasADi
    F_sol_first = cs.DM(F_sol_first)

    #print(type(F_sol_first))
    inputs = cs.vertcat(current_X, F_sol_first)

    # DEBUG: Print inputs
    #print("inputs:", type(inputs), inputs)

    derivatives = learned_dyn(inputs)

    # Simulate dynamics and update the current state
    current_X = current_X + derivatives*dt

    # DEBUG: Print current_X after dynamics update
    print("The current state is: ", current_X)
    
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
    u_history[i,:] = F_sol_first.full().flatten()

print("Plot the results")

# Plotting the trajectory
plt.figure()
plt.plot(x_history, y_history, marker='o')
plt.title('Traiettoria del robot uniciclo')
plt.xlabel('Posizione X [m]')
plt.ylabel('Posizione Y [m]')
plt.grid(True)
plt.axis('scaled')

plt.show()
