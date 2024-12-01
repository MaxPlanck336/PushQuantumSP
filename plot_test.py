import neal
import numpy as np  
from data.sp_data import SPData
from models import SPQuboBinary
import time
import matplotlib.pyplot as plt
from plotting.sp_plot import SPPlot

# Configuration pour LaTeX
plt.rc("text", usetex=False)
plt.rc("font", size=14)  # Taille de la police par défaut
plt.rc("axes", labelsize=18)  # Taille des labels des axes

# Listes pour stocker les données
node_counts = []  # Nombre de nœuds dans le système
sizes_diff = []  # Différence entre la taille de la matrice Q avec et sans preprocessing
times_diff = []  # Différence entre les temps totaux avec et sans preprocessing

# Paramètres à tester

param_list = [{"version": 1, "num_cols": 5, "rad_max": 2.0},
              {"version": 1, "num_cols": 7, "rad_max": 2.4},
              {"version": 1, "num_cols": 10, "rad_max": 3.0},
              {"version": 1, "num_cols": 50, "rad_max": 2.0},
              {"version": 1, "num_cols": 70, "rad_max": 2.4},
              {"version": 1, "num_cols": 5, "rad_max": 2.0},
              {"version": 1, "num_cols": 23, "rad_max": 2.4},
              {"version": 1, "num_cols": 53, "rad_max": 3.3},
              {"version": 1, "num_cols": 80, "rad_max": 2.5},
              {"version": 1, "num_cols": 90, "rad_max": 2.4},
              {"version": 1, "num_cols": 100, "rad_max": 3.0},
              {"version": 1, "num_cols": 100, "rad_max": 2.0},
              {"version": 1, "num_cols": 150, "rad_max": 2.1},
              {"version": 1, "num_cols": 170, "rad_max": 2.6},
              {"version": 1, "num_cols": 190, "rad_max": 2.1},
              {"version": 1, "num_cols": 150, "rad_max": 3.1}]  # Exemple
# Boucle sur les configurations
for idx, params in enumerate(param_list):
    # Générer les données pour chaque configuration
    data = SPData().gen_problem(**params)
    print(f"Processing with parameters (index {idx}): {params}")
    node_counts.append(len(data.G.nodes))  # Récupération du nombre de nœuds

    config = {"num_reads": 1000, "num_sweeps": 1000}
    solve_func = neal.SimulatedAnnealingSampler().sample_qubo

    # Solution avec process=False
    init_time_no_process = time.time()  
    qubo_model_bin_no_process = SPQuboBinary(data, process=False)
    time_qubo_no_process = time.time() - init_time_no_process 
    answer_no_process = qubo_model_bin_no_process.solve(solve_func, **config)
    total_time_no_process = answer_no_process["runtime"]

    # Solution avec process=True
    init_time_with_process = time.time()    
    qubo_model_bin_with_process = SPQuboBinary(data, process=True)
    time_qubo_with_process = time.time() - init_time_with_process   
    answer_with_process = qubo_model_bin_with_process.solve(solve_func, **config)
    total_time_with_process = answer_with_process["runtime"]

    # Calcul de la différence de taille de matrice ou de temps
    sizes_diff.append(qubo_model_bin_no_process.model.shape[0] - qubo_model_bin_with_process.model.shape[0])  # Différence de taille de matrice
    times_diff.append(total_time_no_process - total_time_with_process)  # Différence de temps

# Tracer la courbe : Différence de taille ou de temps vs Nombre de nœuds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Création de deux sous-graphiques

# Premier sous-graphe : Différence de taille de matrice
ax1.plot(node_counts, sizes_diff, label="$\Delta(Q_{M_1}-Q_{M_2})$", marker="o", color="blue", linestyle="None")
ax1.set_xlabel(r"Number N of nodes in the system", fontsize=26)
ax1.set_ylabel(r"Difference in Matrix Size", fontsize=26)
ax1.legend(fontsize=20)
ax1.grid(True)

# Deuxième sous-graphe : Différence de temps
ax2.plot(node_counts, times_diff, label="$\Delta(t_{M_1}-t_{M_2})$[s]", marker="x", color="red", linestyle="None")
ax2.set_xlabel(r"Number N of nodes in the system", fontsize=26)
ax2.set_ylabel(r"Difference in Time (s)", fontsize=26)
ax2.legend(fontsize=20)
ax2.grid(True)

# Afficher la grille et ajuster l'espace
plt.tight_layout()

# Sauvegarder la figure sous forme d'image PNG
plt.savefig("difference_matrix_size_and_time.png")  # Sauvegarder la figure
plt.show()  # Afficher la figure

