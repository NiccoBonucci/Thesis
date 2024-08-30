import torch
import torch.nn as nn
import l4casadi as l4c

import casadi as cs
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
import time
import os

from pylab import *
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


x_start = np.array([0,0,0])         # initial state
x_final   = np.array([4,4,np.pi/2])   # desired terminal state values
f_final   = np.array([0,0])     # desired final control values

# MPC parameters
t_horizon = 5.0
N = 30
steps = 50
dt = t_horizon / N

# Dimensions
nx = 3
nu = 2
ny = nx + nu

class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = torch.nn.Linear(5, 512)

        hidden_layers = []
        for i in range(5):
            hidden_layers.append(torch.nn.Linear(512, 512))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 3)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x
"""    
class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
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

        #print("vo shape:", vo.shape)  # Aggiungi questa riga per il debug
        
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

class UnicycleWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn
        self.parameter_values = self.learned_dyn.get_params(np.array([0,0,0,0,0]))

    def model(self):
        # Definizione delle variabili di stato
        x = cs.MX.sym('x', 1)
        y = cs.MX.sym('y', 1)
        theta = cs.MX.sym('theta', 1)

        # Definizione degli ingressi
        v = cs.MX.sym('v', 1)
        omega = cs.MX.sym('omega', 1)

        states = cs.vertcat(x, y, theta)
        controls = cs.vertcat(v, omega)

        # Input per la rete neurale (stati + ingressi)
        inputs = cs.vertcat(states, controls)

        # Derivate delle variabili di stato
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        theta_dot = cs.MX.sym('theta_dot')
        xdot = cs.vertcat(x_dot, y_dot, theta_dot)

        derivatives = self.learned_dyn(inputs)
        p = self.learned_dyn.get_sym_params()

        # Definizione della dinamica esplicita del sistema
        f_expl = derivatives

        # Definizione della dinamica implicita del sistema
        f_impl = f_expl - xdot

        # Creazione del modello per ACADOS
        model = cs.types.SimpleNamespace()
        model.x = states
        model.xdot = xdot  
        model.u = controls
        model.z = cs.vertcat([])
        model.p = p
        model.f_expl = f_expl
        model.f_impl = f_impl 
        model.cost_y_expr = cs.vertcat(states, controls)
        model.cost_y_expr_e = cs.vertcat(states)
        model.x_start = x_start
        model.x_final = x_final
        model.f_final = f_final
        model.parameter_values = self.parameter_values
        model.constraints = cs.vertcat([])
        model.name = "unicycle_model"

        return model

class MPC:
    def __init__(self, model, N, external_shared_lib_dir, external_shared_lib_name):
        self.N = N
        self.model = model
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property
    def simulator(self):
        return AcadosSimSolver(self.sim())

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def sim(self):
        model = self.model

        t_horizon = 5.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Create OCP object to formulate the optimization
        sim = AcadosSim()
        sim.model = model_ac
        sim.dims.N = N
        sim.dims.nx = nx
        sim.dims.nu = nu
        sim.dims.ny = nx + nu
        sim.solver_options.tf = t_horizon

        # Solver options
        sim.solver_options.Tsim = dt
        sim.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        sim.solver_options.hessian_approx = 'GAUSS_NEWTON'
        sim.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 0
        sim.solver_options.nlp_solver_type = 'SQP_RTI'

        return sim

    def ocp(self):
        model = self.model

        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Initialize cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        """
        # Definizione dei pesi - BEST X
        w_x = 6.45  # Peso per x
        w_y = 0.25  # Peso per y
        w_theta = 10.5  # Peso maggiore per l'orientamento theta
        w_vlin = 1.0  # Peso per l'input di controllo (sforzo minimo)
        w_omega = 1.0  # Peso per l'input di controllo (sforzo minimo)

        ocp.cost.W = np.diag([w_x, w_y, w_theta, w_vlin, w_omega])
        ocp.cost.W_e = np.diag([12.5, 5.0, 15.0])        
        """

        # Definizione dei pesi
        w_x = 1.0  # Peso per x
        w_y = 1.0  # Peso per y
        w_theta = 1.0  # Peso maggiore per l'orientamento theta
        w_vlin = 0.2  # Peso per l'input di controllo (sforzo minimo)
        w_omega = 0.2  # Peso per l'input di controllo (sforzo minimo)

        ocp.cost.W = np.diag([w_x, w_y, w_theta, w_vlin, w_omega])
        ocp.cost.W_e = np.diag([20.0, 10.0, 30.0])        
        
        ocp.cost.Vx = np.zeros((5, 3))
        ocp.cost.Vx[:3, :3] = np.eye(3)
        ocp.cost.Vx_e = np.eye(3)

        ocp.cost.Vu = np.zeros((5, 2))
        ocp.cost.Vu[3:, :] = np.eye(2)

        ocp.cost.Vz = np.array([[]])

        # Initial reference trajectory (will be overwritten)
        ocp.cost.yref = np.array([x_final[0], x_final[1], x_final[2], f_final[0], f_final[1]])
        ocp.cost.yref_e = np.array([x_final[0], x_final[1], x_final[2]])

        """
        ocp.cost.yref_e = np.array([x_final[0], x_final[1], x_final[2]])

        # Initial reference trajectory (will be overwritten)
        ocp.cost.yref = np.array([x_final[0], x_final[1], x_final[2]])
        """

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Initial state (will be overwritten)
        ocp.constraints.x_e = model.x_final
        ocp.constraints.u_e = f_final
        
        """
        ocp.constraints.lbu_e = np.array([f_final[0]-0.1, f_final[1]-0.1])
        ocp.constraints.ubu_e = np.array([f_final[0]+0.1, f_final[1]+0.1])
        ocp.constraints.idxbu_e = np.array([0,1])
        ocp.constraints.lbx_e = np.array([x_final[0]-0.05, x_final[1]-0.05, x_final[2]-0.03])
        ocp.constraints.ubx_e = np.array([x_final[0]+0.05, x_final[1]+0.05, x_final[2]+0.03])
        ocp.constraints.idxbx_e = np.array([0,1,2])        
        """

        # Set constraints
        v_max = 5
        omega_max = 2*np.pi
        ocp.constraints.lbu = np.array([0, -omega_max])
        ocp.constraints.ubu = np.array([v_max, omega_max])
        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.lbx = np.array([-np.pi])
        ocp.constraints.ubx = np.array([np.pi])
        ocp.constraints.idxbx = np.array([2])

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
        ocp.parameter_values = model.parameter_values

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.parameter_values  # Aggiungi questa riga per passare i parametri
        model_ac.name = model.name
        return model_ac



def run():
    
    # Carica il modello addestrato della rete neurale
    PyTorch_model = PyTorchModel()
    PyTorch_model.load_state_dict(torch.load("unicycle_model_state_dense.pth"))
    PyTorch_model.eval()
    learned_dyn_model = l4c.realtime.RealTimeL4CasADi(PyTorch_model, approximation_order=1)
    model = UnicycleWithLearnedDynamics(learned_dyn_model)
    solver = MPC(model=model.model(), N=N,
                 external_shared_lib_dir=learned_dyn_model.shared_lib_dir,
                 external_shared_lib_name=learned_dyn_model.name).solver
    # Definisci il modello dinamico dell'uniciclo con la rete neurale
    """    
    print('Warming up model...')
    x_l = []
    for i in range(N):
        x_l.append(solver.get(i, "x"))
        
    for i in range(20):
        learned_dyn_model.get_params(np.stack(x_l, axis=0))
    print('Warmed up!')
    """

    # Stato iniziale (x, y, theta)
    xt = x_start
    x = [xt]
    
    # Per salvare i controlli ottimali
    u_history = []

    opt_times = []

    for i in range(steps):
        now = time.time()

        # Imposta lo stato finale e l'ingresso finale come riferimento per la funzione di costo
        solver.set(N, "yref", np.hstack((x_final)))

        # Imposta lo stato corrente come vincolo iniziale
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        
        # Risolvi il problema di controllo ottimo
        solver.solve()

        # Recupera la prima azione di controllo ottima (velocità lineare e angolare)
        u_opt = solver.get(0, "u")
        u_history.append(u_opt)

        # Usa le equazioni dell'uniciclo per calcolare il nuovo stato
        dt = t_horizon / N
        v, omega = u_opt
        x_new = xt[0] + v * np.cos(xt[2]) * dt 
        y_new = xt[1] + v * np.sin(xt[2]) * dt 
        theta_new = xt[2] + omega * dt 
        
        theta_new = theta_new - 2 * np.pi * cs.floor((theta_new + np.pi) / (2 * np.pi))

        xt = np.array([x_new, y_new, theta_new])
        x.append(xt)

        # Aggiorna i parametri della rete neurale basati sui nuovi stati
        x_l = []
        for i in range(N):
            x_state = solver.get(i, "x")
            u_control = solver.get(i, "u")
            x_l.append(np.hstack((x_state, u_control)))
        params = learned_dyn_model.get_params(np.stack(x_l, axis=0))
        for i in range(N):
            solver.set(i, "p", params[i])

        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz')

    # Stampa il risultato finale
    print("Final state reached:", xt)

    # Trasforma x e controls in array numpy per la visualizzazione
    x = np.array(x)
    u_history = np.array(u_history)
    
    # Calcola gli errori rispetto allo stato finale desiderato
    error_x = x[:, 0] - x_final[0]
    error_y = x[:, 1] - x_final[1]

    error_theta = x[:, 2] - x_final[2]
    error_theta = error_theta - 2 * np.pi * cs.floor((error_theta + np.pi) / (2 * np.pi))

    print(f"Final error on x: {abs(error_x[-1]):.6f} m")
    print(f"Final error on y: {abs(error_y[-1]):.6f} m")
    print(f"Final error on theta: {abs(error_theta[-1]):.6f} rad")
    
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], label='x(t)')
    plt.title('Traiettoria del robot uniciclo')
    plt.xlabel('Posizione x [m]')
    plt.ylabel('Posizione y [m]')
    plt.grid(True)
    plt.axis('scaled')
    plt.show()

    # Traccia la traiettoria x, y e theta rispetto al tempo
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(x[:, 2], label='theta(t)')
    plt.title('Traiettoria di theta')
    plt.xlabel('Tempo [s]')
    plt.ylabel('theta [rad]')

    plt.subplot(3, 1, 2)
    plt.plot(u_history[:, 0], label='v(t)')
    plt.title('Velocità lineare v')
    plt.xlabel('Tempo (s)')
    plt.ylabel('v [m/s]')

    plt.subplot(3, 1, 3)
    plt.plot(u_history[:, 1], label='omega(t)')
    plt.title('Velocità angolare omega')
    plt.xlabel('Tempo (s)')
    plt.ylabel('omega [rad/s]')

    plt.tight_layout()
    plt.show()
    
    # Traccia gli errori rispetto allo stato finale
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(error_x, label='Errore x(t)')
    plt.title('Errore su x rispetto a posizione finale')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Errore x [m]')

    plt.subplot(3, 1, 2)
    plt.plot(error_y, label='Errore y(t)')
    plt.title('Errore su y rispetto a posizione finale')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Errore y [m]')

    plt.subplot(3, 1, 3)
    plt.plot(error_theta, label='Errore theta(t)')
    plt.title('Errore su theta rispetto a orientamento finale')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Errore theta [rad]')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run()
