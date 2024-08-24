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
Tf    = 10.0                 # control horizon [s]
Nhor  = 50                  # number of control intervals
dt    = Tf/Nhor             # sample time

current_X = vertcat(0,0,0)  # initial state
final_X   = vertcat(3,4,0)    # desired terminal state
final_F   = vertcat(0,0)    # desired terminal state

Nsim  = 40                   # how much samples to simulate
add_noise = False            # enable/disable the measurement noise addition in simulation
add_disturbance = False      # enable/disable the disturbance addition in simulation

############################################################################
############# Definizione della dinamica tramite rete neurale ##############
############################################################################

# RETE NEURALE 1 - SEPARAZIONE STRATI PER VALUTAZIONE ANGOLO ACCURATA

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        # Layer per x e y (con ReLU)
        self.xy_layer = nn.Linear(2, 256)
        
        # Layer per theta (con Tanh)
        self.theta_layer = nn.Linear(1, 512)
        
        # Layer per v e omega (con ReLU)
        self.vo_layer = nn.Linear(2, 256)
        
        # Layer per la combinazione dei tre percorsi
        self.combined_layer = nn.Linear(256 + 512 + 256, 512)
        
        # Hidden layers
        hidden_layers = []
        for i in range(3):
            hidden_layers.append(nn.Linear(512, 512))
        
        self.hidden_layer = nn.ModuleList(hidden_layers)
        
        # Output layer
        self.out_layer = nn.Linear(512, 3)
        
    def forward(self, x):
        # Separazione dei componenti input
        xy = x[:, :2]  # x, y
        theta = x[:, 2:3]  # theta
        vo = x[:, 3:]  # v, omega
        
        # Processamento dei componenti
        xy = torch.relu(self.xy_layer(xy))
        theta = torch.tanh(self.theta_layer(theta))
        vo = torch.relu(self.vo_layer(vo))
        
        # Concatenazione e passaggio al layer combinato
        combined = torch.cat((xy, theta, vo), dim=1)
        combined = torch.relu(self.combined_layer(combined))
        
        # Passaggio attraverso i hidden layers
        for layer in self.hidden_layer:
            combined = torch.relu(layer(combined))
        
        # Output layer
        output = self.out_layer(combined)
        return output  
"""
# RETE NEURALE 1 - SEPARAZIONE STRATI PER VALUTAZIONE ANGOLO ACCURATA

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        # Layer per x e y (con ReLU)
        self.xy_layer = nn.Linear(2, 256)
        
        # Layer per theta (con Tanh)
        self.theta_layer = nn.Linear(1, 256)
        
        # Layer per v e omega (con ReLU)
        self.vo_layer = nn.Linear(2, 256)
        
        # Layer per la combinazione dei tre percorsi
        self.combined_layer = nn.Linear(256 + 256 + 256, 512)
        
        # Hidden layers
        hidden_layers = []
        for i in range(1):
            hidden_layers.append(nn.Linear(512, 512))
        
        self.hidden_layer = nn.ModuleList(hidden_layers)
        
        # Output layer
        self.out_layer = nn.Linear(512, 3)
        
    def forward(self, x):
        # Separazione dei componenti input
        xy = x[:, :2]  # x, y
        theta = x[:, 2:3]  # theta
        vo = x[:, 3:]  # v, omega
        
        # Processamento dei componenti
        xy = torch.relu(self.xy_layer(xy))
        theta = torch.tanh(self.theta_layer(theta))
        vo = torch.relu(self.vo_layer(vo))
        
        # Concatenazione e passaggio al layer combinato
        combined = torch.cat((xy, theta, vo), dim=1)
        combined = torch.relu(self.combined_layer(combined))
        
        # Passaggio attraverso i hidden layers
        for layer in self.hidden_layer:
            combined = torch.relu(layer(combined))
        
        # Output layer
        output = self.out_layer(combined)
        return output  
"""

# Define the dynamic system using the trained model and l4casADi
PyTorch_model = PyTorchModel()
PyTorch_model.load_state_dict(torch.load("unicycle_model_state_off.pth"))
PyTorch_model.eval()
learned_dyn = l4c.L4CasADi(PyTorch_model, model_expects_batch_dim=True, device='cpu')

# -------------------------------
# Logging variables
# -------------------------------
x_history = np.zeros(Nsim+1)
y_history = np.zeros(Nsim+1)
theta_history = np.zeros(Nsim+1)
u_history = np.zeros((Nsim, 2))

error_x = np.zeros(Nsim+1)
error_y = np.zeros(Nsim+1)
error_theta = np.zeros(Nsim+1)
# -------------------------------
# Set OCP
# -------------------------------
ocp = Ocp(T=FreeTime(Tf))

# Define states
x = ocp.state()  # [m]
y = ocp.state()  # [m]
theta  = ocp.state()  # [rad]
X = cs.vertcat(x, y, theta)

# Define controls
V = ocp.control()
omega = ocp.control()
F = cs.vertcat(V, omega)

inputs = cs.vertcat(X, F)
derivatives = learned_dyn(inputs)

# Impostazione delle equazioni differenziali nel problema OCP
ocp.set_der(X, derivatives)

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

ocp.subject_to(0 <= (V<=5))
ocp.subject_to( -5 <= (omega<= 5))
#ocp.subject_to(-5 <= (x <= 5), include_first=False)
#ocp.subject_to(-5 <= (y <= 5), include_first=False)
ocp.subject_to( -np.pi <= (theta<= np.pi), include_first=False)

ocp.solver('ipopt')

method = DirectCollocation(N=Nhor, M=2, intg='rk')
ocp.method(method)

# Definizione dei pesi
w_x = 1.0  # Peso per x
w_y = 1.0  # Peso per y
w_theta = 10.0  # Peso maggiore per l'orientamento theta
w_vlin = 1.0  # Peso per l'input di controllo (sforzo minimo)
w_omega = 3.0  # Peso per l'input di controllo (sforzo minimo)

# Minimal time
#ocp.add_objective(ocp.T)

# Cost Function
ocp.add_objective(ocp.integral(w_x*(x - final_X[0])**2 + w_y*(y - final_X[1])**2 + w_theta*(theta - final_X[2])**2 + w_vlin*(V - final_F[0])**2 + w_omega*(omega - final_F[1])**2))

# Set initial value for parameters
ocp.set_value(X_0, current_X)

# Solve
sol = ocp.solve()

###################################################
############ Grafico soluzione iniziale ###########
###################################################

figure()

ts_initial, xs_initial = sol.sample(x, grid='control')
ts_initial, ys_initial = sol.sample(y, grid='control')

plot(xs_initial, ys_initial,'bo')

ts_initial, xs_initial = sol.sample(x, grid='integrator')
ts_initial, ys_initial = sol.sample(y, grid='integrator')

plot(xs_initial, ys_initial, 'b.')

ts_initial, xs_initial = sol.sample(x, grid='integrator',refine=10)
ts_initial, ys_initial = sol.sample(y, grid='integrator',refine=10)

plot(xs_initial, ys_initial, '-')

axis('equal')
show(block=True)

ts_initial , thetas_initial = sol.sample(theta, grid='control')

plot(ts_initial, thetas_initial,'go')

ts_initial , thetas_initial = sol.sample(theta, grid='integrator')

plot(ts_initial, thetas_initial,'g.')

ts_initial, thetas_initial = sol.sample(theta, grid='integrator', refine=10)

plot(ts_initial, thetas_initial,'-')

axis('equal')
show(block=True)

Sim_unycicle_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
x_history[0]   = current_X[0]
y_history[0]   = current_X[1]
theta_history[0] = current_X[2]

error_x[0] = current_X[0] - final_X[0]
error_y[0] = current_X[1] - final_X[1]
error_theta[0] = current_X[2] - final_X[2]

DM.rng(0)

for i in range(Nsim):
    print("   ")
    print("timestep", i+1, "of", Nsim)
    print("   ")

    # Get the solution from sol
    tsa, Fsol = sol.sample(F, grid='control')
    F_sol_first = Fsol[0, :]

    # DEBUG: Print current_X after dynamics update
    print("The control action to apply is: ", F_sol_first)

    # Convert current_X to DM for compatibility with CasADi
    #F_sol_first = cs.DM(F_sol_first)

    # Simulate dynamics and update the current state
    current_X = Sim_unycicle_dyn(x0=current_X, u=F_sol_first, T=dt)["xf"]

    current_X[2] = current_X[2] - 2 * np.pi * cs.floor((current_X[2] + np.pi) / (2 * np.pi))

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
    u_history[i,:] = F_sol_first

    # Calcolo degli errori
    error_x[i+1] = x_history[i+1] - final_X[0].full()
    error_y[i+1] = y_history[i+1] - final_X[1].full()

    # Calcolo dell'errore su theta con normalizzazione tra -pi e pi
    error_theta[i+1] = theta_history[i+1] - final_X[2].full()
    error_theta[i+1] = np.mod(error_theta[i+1] + np.pi, 2 * np.pi) - np.pi

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

# Definisci il tempo per la storia degli input
time_inputs = np.linspace(0, (Nsim-1)*dt, Nsim)

# Plot dell'input V (velocità lineare)
plt.figure()
plt.plot(time_inputs, u_history[:, 0], marker='o', color='m', label='Input V (velocità lineare)')
plt.title('Evoluzione di V nel tempo')
plt.xlabel('Tempo [s]')
plt.ylabel('V [m/s]')
plt.grid(True)
plt.legend()
plt.show()

# Plot dell'input omega (velocità angolare)
plt.figure()
plt.plot(time_inputs, u_history[:, 1], marker='o', color='c', label='Input Omega (velocità angolare)')
plt.title('Evoluzione di Omega nel tempo')
plt.xlabel('Tempo [s]')
plt.ylabel('Omega [rad/s]')
plt.grid(True)
plt.legend()
plt.show()

# Creazione del vettore temporale per l'asse x
time = np.linspace(0, Nsim*dt, Nsim+1)

# Plot dell'errore su x
plt.figure()
plt.plot(time, error_x, marker='o', color='r', label='Errore su x')
plt.title('Errore su X nel tempo')
plt.xlabel('Tempo [s]')
plt.ylabel('Errore X [m]')
plt.grid(True)
plt.legend()
plt.show()

# Plot dell'errore su y
plt.figure()
plt.plot(time, error_y, marker='o', color='g', label='Errore su y')
plt.title('Errore su Y nel tempo')
plt.xlabel('Tempo [s]')
plt.ylabel('Errore Y [m]')
plt.grid(True)
plt.legend()
plt.show()

# Plot dell'errore su theta
plt.figure()
plt.plot(time, error_theta, marker='o', color='b', label='Errore su theta')
plt.title('Errore su Theta nel tempo')
plt.xlabel('Tempo [s]')
plt.ylabel('Errore Theta [rad]')
plt.grid(True)
plt.legend()
plt.show()

