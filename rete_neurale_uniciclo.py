import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import matplotlib.pyplot as plt

# Definizione del modello MLP
class UnicycleDynamicsNN(nn.Module):
    def __init__(self):
        super(UnicycleDynamicsNN, self).__init__()
        self.fc1 = nn.Linear(5, 512)  # 5 input nodes: x, y, theta, v, omega
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)  # 3 output nodes: x_next, y_next, theta_next
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_unicycle_data(num_samples=10000):
    # Definizione degli array vuoti per input e output
    X = np.zeros((num_samples, 5))
    Y = np.zeros((num_samples, 3))
    
    for i in range(num_samples):
        # Stato casuale (x, y, theta)
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        theta = np.random.uniform(-np.pi, np.pi)
        
        # Comandi casuali (v, omega)
        v = np.random.uniform(0, 2)
        omega = np.random.uniform(-np.pi, np.pi)
        
        # Equazioni differenziali dell'uniciclo
        dx_dt = v * np.cos(theta)
        dy_dt = v * np.sin(theta)
        dtheta_dt = omega
        
        # Popolamento del dataset
        X[i, :] = [x, y, theta, v, omega]
        Y[i, :] = [dx_dt, dy_dt, dtheta_dt]
    
    return X, Y

# Genera il dataset
X, Y = generate_unicycle_data()

# Conversione in tensori di PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Creazione del dataset PyTorch
dataset = TensorDataset(X_tensor, Y_tensor)

# Divisione in train e test set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Creazione del DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inizializza la rete, il criterio di perdita e l'ottimizzatore
model = UnicycleDynamicsNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop di addestramento
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Addestramento completato')

torch.save(model.state_dict(), 'unicycle_model_state.pth')

model.eval()  # Mette la rete in modalità valutazione
test_loss = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f'Test Loss: {test_loss/len(test_loader):.4f}')

# Esempio di confronto
def compare_model_with_exact(model, X_sample):
    with torch.no_grad():
        predicted = model(X_sample)
    
    # Equazioni differenziali esatte
    x, y, theta, v, omega = X_sample[0].numpy()
    true_dx_dt = v * np.cos(theta)
    true_dy_dt = v * np.sin(theta)
    true_dtheta_dt = omega
    
    true_values = np.array([true_dx_dt, true_dy_dt, true_dtheta_dt])
    
    print(f'Predicted: {predicted.numpy()}')
    print(f'True: {true_values}')

# Prendi un campione casuale dal test set
sample_index = np.random.randint(0, len(test_dataset))
X_sample, Y_sample = test_dataset[sample_index]

compare_model_with_exact(model, X_sample.unsqueeze(0))

def plot_trajectory(model, initial_state, dt, steps=50):
    trajectory = np.zeros((steps, 3))
    trajectory[0] = initial_state[:3]  # (x, y, theta)
    
    state = torch.tensor(initial_state, dtype=torch.float32)
    
    for i in range(1, steps):
        with torch.no_grad():
            print(type(state.unsqueeze(0)))
            derivatives = model(state.unsqueeze(0)).squeeze(0).numpy()

        state[:3] = state[:3] + derivatives * dt  # x_next = x_current + dx/dt * dt (dt=1 per semplicità)
        trajectory[i] = state[:3]
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
    plt.title('Trajectory Predicted by the Neural Network')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('scaled')
    plt.show()

# Esegui la simulazione e plottala
initial_state = np.array([0.0, 0.0, 0.0, 1.0, 0.1])  # (x, y, theta, v, omega)
plot_trajectory(model, initial_state, 0.1)
