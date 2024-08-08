import torch
import torch.nn as nn
import l4casadi as l4c

from casadi import *
import casadi as cs

from rockit import *

import numpy as np
import matplotlib.pyplot as plt

from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time

COST = 'LINEAR_LS'  # NONLINEAR_LS

class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = torch.nn.Linear(5, 512)

        hidden_layers = []
        for i in range(5):
            hidden_layers.append(torch.nn.Linear(512, 512))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 3)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.out_layer.bias.fill_(0.)
            self.out_layer.weight.fill_(0.)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.relu(layer(x))
        x = self.out_layer(x)
        return x

class UnicycleWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        x = cs.MX.sym('x', 1)
        y = cs.MX.sym('y', 1)
        theta = cs.MX.sym('theta', 1)
        v = cs.MX.sym('v', 1)
        omega = cs.MX.sym('omega', 1)
        
        # Stato
        state = cs.vertcat(x, y, theta)
        # Controlli
        controls = cs.vertcat(v, omega)
        
        # Stati uniti ai controlli per passare alla rete neurale
        state_with_controls = cs.vertcat(state, controls)

        # Risultato del modello appreso
        res_model = self.learned_dyn(state_with_controls)

        dx = res_model[0]
        dy = res_model[1]
        dtheta = res_model[2]

        x_start = np.zeros((3, ))  # Stato iniziale corretto

        model = cs.types.SimpleNamespace()
        model.x = state
        model.xdot = cs.vertcat(dx, dy, dtheta)
        model.u = controls
        model.z = cs.vertcat([])
        model.p = cs.vertcat([])
        model.f_expl = cs.vertcat(dx, dy, dtheta)
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "unicycle"

        return model


class MPC:
    def __init__(self, model, N, external_shared_lib_dir, external_shared_lib_name):
        self.N = N
        self.model = model
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def ocp(self):
        model = self.model

        t_horizon = 1.0
        N = self.N

        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        nx = 3  # 3 state variables (x, y, theta)
        nu = 2  # 2 control inputs (v, omega)
        ny = 3  # 3 outputs (dx, dy, dtheta)

        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        if COST == 'LINEAR_LS':
            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'

            ocp.cost.W = np.eye(ny)

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[0, 0] = 1.0
            ocp.cost.Vx[1, 1] = 1.0
            ocp.cost.Vx[2, 2] = 1.0
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vz = np.array([[]])
            ocp.cost.Vx_e = np.zeros((ny, nx))

            l4c_y_expr = None
        else:
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.cost.cost_type_e = 'NONLINEAR_LS'

            x = ocp.model.x

            ocp.cost.W = np.eye(ny)

            l4c_y_expr = l4c.L4CasADi(lambda x: x[0], name='y_expr')

            ocp.model.cost_y_expr = l4c_y_expr(x)
            ocp.model.cost_y_expr_e = x[0]

        ocp.cost.W_e = np.zeros((ny, ny))
        ocp.cost.yref_e = np.zeros(ny)

        ocp.cost.yref = np.zeros(ny)

        ocp.constraints.x0 = model.x_start

        v_max = 1.0  # Maximum linear velocity
        omega_max = np.pi / 4  # Maximum angular velocity
        ocp.constraints.lbu = np.array([-v_max, -omega_max])
        ocp.constraints.ubu = np.array([v_max, omega_max])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        if COST == 'LINEAR_LS':
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
        else:
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name + ' -l' + l4c_y_expr.name

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac

def run():
    N_hor = 10
    t_hor = 2.0
    learned_dyn_model = l4c.L4CasADi(PyTorchModel(), model_expects_batch_dim=True, name='learned_dyn')

    model = UnicycleWithLearnedDynamics(learned_dyn_model)
    solver = MPC(model=model.model(), N=N_hor,
                 external_shared_lib_dir=learned_dyn_model.shared_lib_dir,
                 external_shared_lib_name=learned_dyn_model.name).solver

    x = []
    x_ref = []

    ts = t_hor / N_hor
    xt = np.array([0.0, 0.0, 0.0])  # Initial state: x, y, theta
    x_goal = np.array([5.0, 5.0, np.pi / 4])  # Goal state: x, y, theta
    opt_times = []

    for i in range(50):
        now = time.time()
        
        # Imposta yref come lo stato finale per ogni istante temporale
        yref = np.tile(x_goal, (N_hor, 1)).T

        for t_idx in range(N_hor):
            solver.set(t_idx, "yref", yref[:, t_idx])
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        solver.solve()
        
        # Ottieni i primi 3 stati (x, y, theta)
        xt = solver.get(1, "x")[:3]
        x.append(xt)

        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time: {1000 * np.mean(opt_times):.1f} ms -- {1 / np.mean(opt_times):.0f} Hz)')

    # Plot the results
    x = np.array(x)
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], label='Traiettoria')
    plt.plot(x_goal[0], x_goal[1], 'ro', label='Stato Finale')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Traiettoria dell\'Uniciclo')
    plt.legend()
    plt.grid()
    plt.show()

    # Controlli v e omega
    v = [solver.get(i, 'u')[0] for i in range(N_hor)]
    omega = [solver.get(i, 'u')[1] for i in range(N_hor)]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(v)
    plt.ylabel('v')
    plt.grid()
    plt.title('Controlli nel tempo')
    
    plt.subplot(2, 1, 2)
    plt.plot(omega)
    plt.ylabel('omega')
    plt.xlabel('Tempo (passi)')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run()




"""
import torch
import torch.nn as nn
import l4casadi as l4c

from casadi import *
import casadi as cs

from rockit import *

import numpy as np
import matplotlib.pyplot as plt

from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time

COST = 'LINEAR_LS'  # NONLINEAR_LS

class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = torch.nn.Linear(5, 512)

        hidden_layers = []
        for i in range(5):
            hidden_layers.append(torch.nn.Linear(512, 512))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 3)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.out_layer.bias.fill_(0.)
            self.out_layer.weight.fill_(0.)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.relu(layer(x))
        x = self.out_layer(x)
        return x

class UnicycleWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        x = cs.MX.sym('x', 1)
        y = cs.MX.sym('y', 1)
        theta = cs.MX.sym('theta', 1)
        v = cs.MX.sym('v', 1)
        omega = cs.MX.sym('omega', 1)
        
        # Stato
        state = cs.vertcat(x, y, theta)
        # Controlli
        controls = cs.vertcat(v, omega)
        
        # Stati uniti ai controlli per passare alla rete neurale
        state_with_controls = cs.vertcat(state, controls)

        # Risultato del modello appreso
        res_model = self.learned_dyn(state_with_controls)

        dx = res_model[0]
        dy = res_model[1]
        dtheta = res_model[2]

        x_start = np.zeros((3, ))  # Stato iniziale corretto

        model = cs.types.SimpleNamespace()
        model.x = state
        model.xdot = cs.vertcat(dx, dy, dtheta)
        model.u = controls
        model.z = cs.vertcat([])
        model.p = cs.vertcat([])
        model.f_expl = cs.vertcat(dx, dy, dtheta)
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "unicycle"

        return model


class MPC:
    def __init__(self, model, N, external_shared_lib_dir, external_shared_lib_name):
        self.N = N
        self.model = model
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def ocp(self):
        model = self.model

        t_horizon = 1.0
        N = self.N

        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        nx = 3  # 3 state variables (x, y, theta)
        nu = 2  # 2 control inputs (v, omega)
        ny = 3  # 3 outputs (dx, dy, dtheta)

        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        if COST == 'LINEAR_LS':
            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'

            ocp.cost.W = np.eye(ny)

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[0, 0] = 1.0
            ocp.cost.Vx[1, 1] = 1.0
            ocp.cost.Vx[2, 2] = 1.0
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vz = np.array([[]])
            ocp.cost.Vx_e = np.zeros((ny, nx))

            l4c_y_expr = None
        else:
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.cost.cost_type_e = 'NONLINEAR_LS'

            x = ocp.model.x

            ocp.cost.W = np.eye(ny)

            l4c_y_expr = l4c.L4CasADi(lambda x: x[0], name='y_expr')

            ocp.model.cost_y_expr = l4c_y_expr(x)
            ocp.model.cost_y_expr_e = x[0]

        ocp.cost.W_e = np.zeros((ny, ny))
        ocp.cost.yref_e = np.zeros(ny)

        ocp.cost.yref = np.zeros(ny)

        ocp.constraints.x0 = model.x_start

        v_max = 1.0  # Maximum linear velocity
        omega_max = np.pi / 4  # Maximum angular velocity
        ocp.constraints.lbu = np.array([-v_max, -omega_max])
        ocp.constraints.ubu = np.array([v_max, omega_max])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        if COST == 'LINEAR_LS':
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
        else:
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name + ' -l' + l4c_y_expr.name

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac

def run():
    N_hor = 10
    t_hor = 2.0
    learned_dyn_model = l4c.L4CasADi(PyTorchModel(), model_expects_batch_dim=True, name='learned_dyn')

    model = UnicycleWithLearnedDynamics(learned_dyn_model)
    solver = MPC(model=model.model(), N=N_hor,
                 external_shared_lib_dir=learned_dyn_model.shared_lib_dir,
                 external_shared_lib_name=learned_dyn_model.name).solver

    x = []
    x_ref = []
    ts = t_hor / N_hor
    xt = np.array([0.0, 0.0, 0.0])  # Initial state: x, y, theta
    opt_times = []

    for i in range(50):
        now = time.time()
        t = np.linspace(i * ts, i * ts + ts, 10)
        yref = np.array([np.sin(0.5 * t + np.pi / 2), np.cos(0.5 * t + np.pi / 2), np.zeros_like(t)])
        x_ref.append(yref[:, 0])
        for t_idx, ref in enumerate(yref.T):
            solver.set(t_idx, "yref", ref)
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        solver.solve()
        xt = solver.get(1, "x")[:3]  # Ottieni i primi 3 stati (x, y, theta)
        x.append(xt)

        x_l = []
        for i in range(N_hor):
            x_l.append(solver.get(i, "x")[:3])  # Ottieni i primi 3 stati (x, y, theta)

        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time: {1000 * np.mean(opt_times):.1f} ms -- {1 / np.mean(opt_times):.0f} Hz)')


if __name__ == '__main__':
    run()
"""
