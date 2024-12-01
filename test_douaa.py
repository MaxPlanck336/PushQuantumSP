import neal
import numpy as np  
from data.sp_data import SPData
from models import SPQuboBinary
from evaluation.evaluation import SPEvaluation
from plotting.sp_plot import SPPlot
import networkx as nx
import time
import matplotlib.pyplot as plt

# Définir les configurations des paramètres
params_list = [
    {"version": 3, "num_cols": 7, "rad_max": 2.4},
    {"version": 3, "num_cols": 10, "rad_max": 2.4},
    {"version": 3, "num_cols": 15, "rad_max": 2.4},
    {"version": 2, "num_cols": 7, "rad_max": 2.4},
    {"version": 1, "num_cols": 7, "rad_max": 2.4},
    # Ajouter d'autres configurations si nécessaire
]

# Liste pour stocker les résultats des tests
results = {"size_qubo": [], "time_no_process": [], "time_with_process": []}

config = {"num_reads": 1000, "num_sweeps": 1000}
solve_func = neal.SimulatedAnnealingSampler().sample_qubo

# Boucle sur les différentes configurations de paramètres
for params in params_list:
    # Générer les données pour chaque configuration
    data = SPData().gen_problem(**params)
    print(f"Processing with parameters: {params}")

    # Solution avec process=False
    start_time = time.time()
    qubo_model_bin_no_process = SPQuboBinary(data, process=False)
    time_qubo_no_process = time.time() - start_time
    print(f"Time to compute QUBO with process=False: {time_qubo_no_process:.2f} seconds")
    print("Shape of the QUBO matrix with process=False:", qubo_model_bin_no_process.model.shape)
    results["size_qubo"].append(qubo_model_bin_no_process.model.shape[0])

    # Résolution avec process=False
    start_time = time.time()
    answer_True_no_process = qubo_model_bin_no_process.solve(solve_func, **config)
    time_resolution_no_process = time.time() - start_time
    print(f"Time to solve QUBO with process=False: {time_resolution_no_process:.2f} seconds")
    total_time_no_process = time_qubo_no_process + time_resolution_no_process
    results["time_no_process"].append(total_time_no_process)

    # Solution avec process=True
    start_time = time.time()
    qubo_model_bin_process = SPQuboBinary(data, process=True)
    time_qubo_process = time.time() - start_time
    print(f"Time to compute QUBO with process=True: {time_qubo_process:.2f} seconds")
    print("Shape of the QUBO matrix with process=True:", qubo_model_bin_process.model.shape)

    # Résolution avec process=True
    start_time = time.time()
    answer_True_process = qubo_model_bin_process.solve(solve_func, **config)
    time_resolution_process = time.time() - start_time
    print(f"Time to solve QUBO with process=True: {time_resolution_process:.2f} seconds")
    total_time_process = time_qubo_process + time_resolution_process
    results["time_with_process"].append(total_time_process)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(results["size_qubo"], results["time_no_process"], label="process=False", marker='o')
plt.plot(results["size_qubo"], results["time_with_process"], label="process=True", marker='x')
plt.xlabel("Taille de la matrice QUBO")
plt.ylabel("Temps total (secondes)")
plt.title("Temps de calcul en fonction de la taille de la matrice QUBO")
plt.legend()
plt.grid(True)
plt.show()
