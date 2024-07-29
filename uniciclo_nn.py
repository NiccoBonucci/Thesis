
   ###################################################################
   ############ STEP 1: Simulazione Cinematica Uniciclo ##############
   ###################################################################

import numpy as np
import matplotlib.pyplot as plt

def uniciclo_kinematica(x, y, theta, v, omega, dt):
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = omega

    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt

    return x, y, theta

# Parametri di simulazione
dt = 0.1  # Passo di tempo
t_max = 10  # Tempo totale di simulazione
n_steps = int(t_max / dt)

# Inizializzazione delle variabili
x = 0
y = 0
theta = 0
v = 1  # Velocità costante
omega = 0.1  # Velocità angolare costante

# Liste per memorizzare i risultati
x_traj = []
y_traj = []

# Simulazione
for _ in range(n_steps):
    x, y, theta = uniciclo_kinematica(x, y, theta, v, omega, dt)
    x_traj.append(x)
    y_traj.append(y)

# Visualizzazione della traiettoria
plt.plot(x_traj, y_traj, label='Simulazione')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Traiettoria dell\'uniciclo')
plt.legend()
plt.show()

   ###################################################################
   ######## STEP 2: Costruzione e Addestramento Rete Neurale #########
   ###################################################################

import torch
import torch.nn as nn
import torch.optim as optim

#Input rete neurale: x,y,theta all'istante currente, v lineare al tempo corrente, v angolare al tempo corrente
#Output rete neurale: x,y,theta allo stato successivo

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

"""
# Definizione della rete neurale
class UnicycleNN(nn.Module):
    def __init__(self):
        super(UnicycleNN, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
"""

# Creazione del modello, della funzione di loss e dell'ottimizzatore
model = PyTorchModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generazione dei dati di addestramento
data = []
for _ in range(8000):
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)
    theta = np.random.uniform(-np.pi, np.pi)
    v = np.random.uniform(0, 2)
    omega = np.random.uniform(-1, 1)
    dt = 0.1
    x_next, y_next, theta_next = uniciclo_kinematica(x, y, theta, v, omega, dt)
    data.append((x, y, theta, v, omega, x_next - x, y_next - y, theta_next - theta))

# Convertire i dati in tensori PyTorch
inputs = torch.tensor([[d[0], d[1], d[2], d[3], d[4]] for d in data], dtype=torch.float32)
targets = torch.tensor([[d[5], d[6], d[7]] for d in data], dtype=torch.float32)

   ###################################################################
   ################# STEP 3: Addestramento Modello ###################
   ###################################################################


n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

print('Addestramento completato')

torch.save(model.state_dict(), 'unicycle_model_state.pth')

   ###################################################################
   ################## STEP 4: Validazione Modello ####################
   ###################################################################

# Dati di test
x_test = np.random.uniform(-10, 10)
y_test = np.random.uniform(-10, 10)
theta_test = np.random.uniform(-np.pi, np.pi)
v_test = np.random.uniform(0, 2)
omega_test = np.random.uniform(-1, 1)

# Convertire i dati di test in tensori
input_test = torch.tensor([x_test, y_test, theta_test, v_test, omega_test], dtype=torch.float32)

# Predire con il modello
model.eval()

with torch.no_grad():
    prediction = model(input_test.unsqueeze(0))

print('Predizione:', prediction)

   ###################################################################
   ########## STEP 5: Confronto tra traiettorie uniciclo #############
   ###################################################################

def generate_trajectory_nn(model, x_init, y_init, theta_init, v, omega, dt, n_steps):
    x_traj = [x_init]
    y_traj = [y_init]
    theta = theta_init

    for _ in range(n_steps):
        input_vector = torch.tensor([x_traj[-1], y_traj[-1], theta, v, omega], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            delta = model(input_vector.unsqueeze(0)).numpy()[0]
        x_next, y_next, theta_next = x_traj[-1] + delta[0], y_traj[-1] + delta[1], theta + delta[2]
        x_traj.append(x_next)
        y_traj.append(y_next)
        theta = theta_next

    return x_traj, y_traj

# Generazione della traiettoria reale usando la funzione cinematica
def generate_trajectory_real(x_init, y_init, theta_init, v, omega, dt, n_steps):
    x_traj = [x_init]
    y_traj = [y_init]
    theta = theta_init

    for _ in range(n_steps):
        x_next, y_next, theta_next = uniciclo_kinematica(x_traj[-1], y_traj[-1], theta, v, omega, dt)
        x_traj.append(x_next)
        y_traj.append(y_next)
        theta = theta_next

    return x_traj, y_traj

# Parametri di simulazione
x_init, y_init, theta_init = 0, 0, 0
v = 1
omega = 0.1
dt = 0.1
n_steps = 100

# Generazione delle traiettorie
x_traj_nn, y_traj_nn = generate_trajectory_nn(model, x_init, y_init, theta_init, v, omega, dt, n_steps)
x_traj_real, y_traj_real = generate_trajectory_real(x_init, y_init, theta_init, v, omega, dt, n_steps)

# Visualizzazione delle traiettorie
plt.plot(x_traj_nn, y_traj_nn, label='NN Predicted')
plt.plot(x_traj_real, y_traj_real, label='Real')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Traiettoria dell\'uniciclo')
plt.legend()
plt.show()
