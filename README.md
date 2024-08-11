
Questo Github contiene tutti i file realtivi alla Tesi Magistrale di Niccolò Bonucci

Regarding the files uploaded, here's a brief description of each one of them:
- *rete_neurale_uniciclo.py* : codice Python per definire, addestrare e validare una rete neurale multipercettrone tramite Pytorch per la previsione delle equazioni che descrivono la dinamica dell'unicilo
- *unicycle_mpc.py* :  codice Python per risolvere un problema di MPC da una posizione iniziale ad una posizione finale sfruttando la dinamica dell'uniciclo come modello di riferimento per il problema. Il framework utilizzato è Rockit
- *unicycle_neuralmpc.py* :  codice Python per risolvere un problema di MPC da una posizione iniziale ad una posizione finale sfruttando la dinamica dell'uniciclo **generata dalla rete neurale** come modello di riferimento per il problema. I framework utilizzati sono Rockit e L4casADi.
- *motion_planning_unicycle.py* : codice Python per risolvere un singolo problema di controllo ottimo da una posizione iniziale ad una posizione finale sfruttando la dinamica dell'uniciclo come modello di riferimento per il problema
