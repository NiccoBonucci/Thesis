
Questo Github contiene tutti i file realtivi alla Tesi Magistrale di Niccolò Bonucci

Regarding the files uploaded, here's a brief description of each one of them:
- Folder "NeuralMPCsimple" : contains files related to the application of a trained simple 1-hidden layer neural network to a Neural MPC problem
- Folder "Rockit_code" : contains all the code developed using the Rockit Framework
- Folder "Acados" : contains the neural MPC code exploiting the Acados framework as the solver
- *rete_neurale_uniciclo_base.py* : codice Python per definire, addestrare e validare una rete neurale multipercettrone con 3 strati nascosti tramite Pytorch per la previsione delle equazioni che descrivono la dinamica dell'uniciclo
- *rete_neurale_uniciclo_var_stato_separate.py* : codice Python per definire, addestrare e validare una rete neurale multipercettrone con strati di input separati per posizione e orientamentoe e 5 strati nascosti sfruttando Pytorch per la previsione delle equazioni che descrivono la dinamica dell'uniciclo
