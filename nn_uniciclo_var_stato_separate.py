import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt

# RETE NEURALE 1 - RETE STANDARD CON 3 STRATI NASCOSTI
"""
class UnicycleDynamicsNN(nn.Module):
    def __init__(self):
        super(UnicycleDynamicsNN, self).__init__()
        self.fc1 = nn.Linear(5, 256)  # 5 input nodes: x, y, theta, v, omega
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 3)  # 3 output nodes: x_next, y_next, theta_next
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x
"""
# RETE NEURALE 1 - SEPARAZIONE STRATI PER VALUTAZIONE ANGOLO ACCURATA
"""
class UnicycleDynamicsNN(nn.Module):

    def __init__(self):
        super(UnicycleDynamicsNN, self).__init__()
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
# RETE NEURALE 2 - SEPARAZIONE STRATI PER VALUTAZIONE ANGOLO ACCURATA

class UnicycleDynamicsNN(nn.Module):

    def __init__(self):
        super(UnicycleDynamicsNN, self).__init__()
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
    
def generate_unicycle_data(num_samples = 100000):
    # Definizione degli array vuoti per input e output
    X = np.zeros((num_samples, 5))
    Y = np.zeros((num_samples, 3))
    
    for i in range(num_samples):
        # Stato casuale (x, y, theta)
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        theta = np.random.uniform(-np.pi, np.pi)
        
        # Dividi il dataset in tre gruppi:
        # 1. Solo velocità lineare
        # 2. Solo velocità angolare
        # 3. Combinazione di velocità lineare e angolare
        
        if i < num_samples * 0.4:
            # Combinazione di velocità lineare e angolare
            v = np.random.uniform(0, 2)
            omega = np.random.uniform(-2*np.pi, 2*np.pi)
        elif (i > num_samples * 0.4) and (i < num_samples * 0.55):
            # Solo velocità lineare
            v = 0.0
            omega = 0.0
        elif (i > num_samples * 0.55) and (i < num_samples * 0.75):
            # Solo velocità lineare
            v = np.random.uniform(0, 5)
            omega = 0.0
        else :
            # Solo velocità angolare
            v = 0.0
            omega = np.random.uniform(-2*np.pi, 2*np.pi)
        
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

#print(X)
#print(Y)

# Conversione in tensori di PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Creazione del dataset PyTorch
dataset = TensorDataset(X_tensor, Y_tensor)

# Divisione in train e test set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Creazione del DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)


# Correzione della funzione di perdita con pesi
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.weights * (outputs - targets) ** 2
        return loss.mean(dim=0).sum()

# Inizializzazione del modello, della loss function e dell'optimizer
model = UnicycleDynamicsNN()
weights = torch.tensor([1.0, 1.0, 10.0])  # Pesi per x, y, theta
criterion = WeightedMSELoss(weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Configurazione del plot in tempo reale
plt.ion()  # Modalità interattiva
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
line, = ax.plot([], [], label='Train Loss')
ax.legend()

train_losses = []
val_losses = []
best_val_loss = float('inf')

def update_plot(epoch, train_loss, val_loss):
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    line.set_xdata(np.arange(len(train_losses)))
    line.set_ydata(train_losses)
    
    ax.relim()
    ax.autoscale_view()
    
    plt.draw()
    plt.pause(0.01)


# Loop di addestramento con validazione e salvataggio del miglior modello
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calcolo della perdita sul validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(torch.float32), val_targets.to(torch.float32)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss)  # Aggiorna il learning rate se necessario

    train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    update_plot(epoch, train_loss, val_loss)
    
    # Salva il modello se ottiene una migliore performance sulla validazione
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_unicycle_model.pth')

print('Addestramento completato')

torch.save(model.state_dict(), 'unicycle_model_state_off.pth')

plt.ioff()
plt.show()

##################################################################
################## Addestramento Rete Neurale ####################
##################################################################

# Grafico Loss in tempo reale
"""
# Lista per salvare la loss dell'addestramento per ogni epoca
train_losses = []

# Configurazione del plot in tempo reale
plt.ion()  # Modalità interattiva
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
line, = ax.plot([], [], label='Train Loss')
ax.legend()

# Funzione per aggiornare il plot
def update_plot(epoch, train_loss):
    train_losses.append(train_loss)
    
    line.set_xdata(np.arange(len(train_losses)))
    line.set_ydata(train_losses)
    
    ax.relim()
    ax.autoscale_view()
    
    plt.draw()
    plt.pause(0.01)

# Loop di addestramento
num_epochs = 150

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
    
    # Stampa la loss e aggiorna il plot
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    update_plot(epoch, running_loss)

print('Addestramento completato')

torch.save(model.state_dict(), 'unicycle_model_state_off.pth')
"""
##################################################################
################### Validazione Rete Neurale #####################
##################################################################

#model.load_state_dict(torch.load("unicycle_model_state_off.pth"))

model.eval()  # Mette la rete in modalità valutazione
test_loss = 0.0

with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f'Test Loss: {test_loss/len(val_loader):.4f}')

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
    
    error = np.abs(true_values - predicted.numpy())

    print(f'Predicted: {predicted.numpy()}')
    print(f'True: {true_values}')
    print(f'Prediction Error: {error}')

# Prendi un campione casuale dal test set
sample_index = np.random.randint(0, len(val_dataset))
X_sample, Y_sample = val_dataset[sample_index]

def plot_trajectory(model, initial_state, dt, steps):
    trajectory = np.zeros((steps, 3))
    trajectory[0] = initial_state[:3]  # (x, y, theta)
    
    state = torch.tensor(initial_state, dtype=torch.float32)
    
    for i in range(1, steps):
        with torch.no_grad():
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

def calculate_true_trajectory(initial_state, dt, steps):
    trajectory = np.zeros((steps, 3))
    trajectory[0] = initial_state[:3]  # (x, y, theta)
    
    state = initial_state.copy()
    
    for i in range(1, steps):
        x, y, theta, v, omega = state
        
        # Equazioni dinamiche dell'uniciclo
        dx_dt = v * np.cos(theta)
        dy_dt = v * np.sin(theta)
        dtheta_dt = omega
        
        state[0] += dx_dt * dt
        state[1] += dy_dt * dt
        state[2] += dtheta_dt * dt
        
        # Normalizzazione di theta tra -π e π
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        
        trajectory[i] = state[:3]
    
    return trajectory

def normalize_angle(angle):
    #Normalizza l'angolo nell'intervallo [-π, π]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def plot_trajectory_comparison(model, initial_state, dt, steps):
    # Traiettoria predetta dalla rete neurale
    predicted_trajectory = np.zeros((steps, 3))
    predicted_trajectory[0] = initial_state[:3]
    
    state = torch.tensor(initial_state, dtype=torch.float32)
    
    for i in range(1, steps):
        with torch.no_grad():
            derivatives = model(state.unsqueeze(0)).squeeze(0).numpy()

        state[:3] = state[:3] + derivatives * dt
        
        # Normalizzazione di theta tra -π e π
        state[2] = normalize_angle(state[2])
        
        predicted_trajectory[i] = state[:3]
    
    # Traiettoria reale calcolata dalle equazioni dinamiche
    true_trajectory = calculate_true_trajectory(initial_state, dt, steps)
    
    # Calcolo dell'errore per ciascuna variabile di stato
    error_x = np.abs(predicted_trajectory[:, 0] - true_trajectory[:, 0])
    error_y = np.abs(predicted_trajectory[:, 1] - true_trajectory[:, 1])
    
    # Calcolo dell'errore angolare considerando la periodicità
    error_theta = normalize_angle(predicted_trajectory[:, 2] - true_trajectory[:, 2])
    error_theta = np.abs(error_theta)  # Assoluto dell'errore angolare
    
    # Asse temporale
    time = np.arange(steps) * dt
    
    # Plot di X
    plt.figure(figsize=(10, 6))
    plt.plot(time, predicted_trajectory[:, 0], marker='o', label='Predicted X')
    plt.plot(time, true_trajectory[:, 0], marker='x', label='True X')
    plt.title('X Position over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('x [m]')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot di Y
    plt.figure(figsize=(10, 6))
    plt.plot(time, predicted_trajectory[:, 1], marker='o', label='Predicted Y')
    plt.plot(time, true_trajectory[:, 1], marker='x', label='True Y')
    plt.title('Y Position over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot di Theta
    plt.figure(figsize=(10, 6))
    plt.plot(time, predicted_trajectory[:, 2], marker='o', label='Predicted Theta')
    plt.plot(time, true_trajectory[:, 2], marker='x', label='True Theta')
    plt.title('Theta Angle over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Theta [rad]')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot degli errori per ogni variabile di stato
    plt.figure(figsize=(10, 6))
    plt.plot(time, error_x, marker='x', label='Error in X')
    plt.plot(time, error_y, marker='x', label='Error in Y')
    plt.plot(time, error_theta, marker='x', label='Error in Theta')
    plt.title('Error in X, Y, and Theta over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m, rad]')
    plt.grid(True)
    plt.legend()
    plt.show()

# Esegui la simulazione e plottala
initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # (x, y, theta, v, omega)
compare_model_with_exact(model, X_sample.unsqueeze(0))
#plot_trajectory(model, initial_state, 0.1, 500)
plot_trajectory_comparison(model, initial_state,0.1,500)
