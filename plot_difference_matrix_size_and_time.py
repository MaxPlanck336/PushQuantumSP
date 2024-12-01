import neal
import numpy as np
from data.sp_data import SPData
from models import SPQuboBinary
import time
import matplotlib.pyplot as plt
from plotting.sp_plot import SPPlot

# Configuration for LaTeX-style plots (optional)
plt.rc("text", usetex=False)  # Disable LaTeX rendering in text
plt.rc("font", size=14)       # Set default font size
plt.rc("axes", labelsize=18)  # Set axis label font size

# Lists to store results
node_counts = []  # Number of nodes in the system
sizes_diff = []   # Difference in matrix size with and without preprocessing
times_diff = []   # Difference in total runtime with and without preprocessing

# Define parameters for testing
param_list = [
    {"version": 1, "num_cols": 5, "rad_max": 2.0},
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
    {"version": 1, "num_cols": 150, "rad_max": 3.1}
]

# Loop through each parameter configuration
for idx, params in enumerate(param_list):
    # Generate problem data for the given configuration
    data = SPData().gen_problem(**params)
    print(f"Processing with parameters (index {idx}): {params}")
    node_counts.append(len(data.G.nodes))  # Record the number of nodes in the graph

    # Solver configuration
    config = {"num_reads": 1000, "num_sweeps": 1000}
    solve_func = neal.SimulatedAnnealingSampler().sample_qubo

    # Solve without preprocessing (process=False)
    init_time_no_process = time.time()  # Start time
    qubo_model_bin_no_process = SPQuboBinary(data, process=False)
    time_qubo_no_process = time.time() - init_time_no_process  # Time taken to build the matrix Q
    answer_no_process = qubo_model_bin_no_process.solve(solve_func, **config)
    total_time_no_process = answer_no_process["runtime"] +time_qubo_no_process # Total runtime

    # Solve with preprocessing (process=True)
    init_time_with_process = time.time()  # Start time
    qubo_model_bin_with_process = SPQuboBinary(data, process=True)
    time_qubo_with_process = time.time() - init_time_with_process  # Time taken to build the matrix Q
    answer_with_process = qubo_model_bin_with_process.solve(solve_func, **config)
    total_time_with_process = answer_with_process["runtime"]+ time_qubo_with_process # Total runtime

    # Calculate the difference in matrix size and runtime
    sizes_diff.append(
        qubo_model_bin_no_process.model.shape[0] - qubo_model_bin_with_process.model.shape[0]
    )  # Difference in QUBO matrix size
    times_diff.append(total_time_no_process - total_time_with_process)  # Difference in runtime

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  

# First subplot: Difference in matrix size
ax1.plot(node_counts, sizes_diff, label="$\Delta(Q_{M_1}-Q_{M_2})$", 
         marker="o", color="blue", linestyle="None")
ax1.set_xlabel(r"Number N of nodes in the system", fontsize=26)
ax1.set_ylabel(r"Difference in Matrix Size", fontsize=26)
ax1.legend(fontsize=20)
ax1.grid(True)

# Second subplot: Difference in runtime
ax2.plot(node_counts, times_diff, label="$\Delta(t_{M_1}-t_{M_2})$[s]", 
         marker="x", color="red", linestyle="None")
ax2.set_xlabel(r"Number N of nodes in the system", fontsize=26)
ax2.set_ylabel(r"Difference in Time (s)", fontsize=26)
ax2.legend(fontsize=20)
ax2.grid(True)

plt.tight_layout()

plt.savefig("difference_matrix_size_and_time.png") 
plt.show() 
