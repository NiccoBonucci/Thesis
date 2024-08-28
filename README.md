
Questo Github contiene tutti i file realtivi alla Tesi Magistrale di Niccolò Bonucci

Regarding the files uploaded, here's a brief description of each one of them:
- Folder "NeuralMPCsimple" contains files related to the application of a trained simple 1-hidden layer neural network to a Neural MPC problem
- *rete_neurale_uniciclo.py* : codice Python per definire, addestrare e validare una rete neurale multipercettrone tramite Pytorch per la previsione delle equazioni che descrivono la dinamica dell'uniciclo
- *motion_planning_unicycle.py* : codice Python per risolvere un singolo problema di controllo ottimo da una posizione iniziale ad una posizione finale sfruttando la dinamica dell'uniciclo come modello di riferimento per il problema
- *unicycle_mpc.py* :  codice Python per risolvere un problema di MPC da una posizione iniziale ad una posizione finale sfruttando la dinamica dell'uniciclo come modello di riferimento per il problema. Il framework utilizzato è Rockit
- *unicycle_neuralmpc.py* :  codice Python per risolvere un problema di MPC da una posizione iniziale ad una posizione finale sfruttando la dinamica dell'uniciclo **generata dalla rete neurale** come modello di riferimento per il problema. I framework utilizzati sono Rockit e L4casADi
- *file.pth* : file necessari per caricare la rete neurale addestrata e utilizzarla nel neural MPC
