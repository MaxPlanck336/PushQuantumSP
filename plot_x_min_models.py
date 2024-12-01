import neal
import numpy as np  
from data.sp_data import SPData
from models import SPQuboBinary
from evaluation.evaluation import SPEvaluation
from plotting.sp_plot import SPPlot
import networkx as nx
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define multiple parameter sets to test
"""
params_list = [
    {"version": 2, "num_cols": 10, "rad_max": 2.2},
    {"version": 3, "num_cols": 4, "rad_max": 3.2},
    {"version": 2, "num_cols": 50, "rad_max": 2.8},
    {"version": 1, "num_cols": 24, "rad_max": 2.2},
    {"version": 3, "num_cols": 43, "rad_max": 2.23},
    {"version": 2, "num_cols": 120, "rad_max": 2.5},
    {"version": 1, "num_cols": 80, "rad_max": 2.9},
    {"version": 3, "num_cols": 70, "rad_max": 3.0},
    {"version": 2, "num_cols": 200, "rad_max": 2.3},
    {"version": 1, "num_cols": 40, "rad_max": 2.0},
    {"version": 3, "num_cols": 30, "rad_max": 3.1},
    {"version": 2, "num_cols": 60, "rad_max": 2.7},
    {"version": 1, "num_cols": 90, "rad_max": 2.1},
    {"version": 3, "num_cols": 110, "rad_max": 2.4},
    {"version": 2, "num_cols": 130, "rad_max": 2.9},
    {"version": 1, "num_cols": 150, "rad_max": 2.8},
    {"version": 3, "num_cols": 180, "rad_max": 2.6},
    {"version": 2, "num_cols": 160, "rad_max": 3.0},
    {"version": 1, "num_cols": 140, "rad_max": 2.1},
    {"version": 3, "num_cols": 190, "rad_max": 3.3}
]
"""
params_list = [{"version": 1, "num_cols": 5, "rad_max": 2.0},
              {"version": 1, "num_cols": 7, "rad_max": 2.4},
              {"version": 1, "num_cols": 10, "rad_max": 3.0},
              {"version": 1, "num_cols": 50, "rad_max": 2.0},
              {"version": 1, "num_cols": 70, "rad_max": 2.4},
              {"version": 1, "num_cols": 5, "rad_max": 2.0},
              {"version": 1, "num_cols": 23, "rad_max": 2.4},
              {"version": 1, "num_cols": 53, "rad_max": 4.3},
              {"version": 1, "num_cols": 80, "rad_max": 2.5},
              {"version": 1, "num_cols": 90, "rad_max": 2.4},
              {"version": 1, "num_cols": 100, "rad_max": 3.0},
              {"version": 1, "num_cols": 100, "rad_max": 2.0},
              {"version": 1, "num_cols": 150, "rad_max": 2.1},
              {"version": 1, "num_cols": 170, "rad_max": 2.6},
              {"version": 1, "num_cols": 190, "rad_max": 2.1},
              {"version": 1, "num_cols": 150, "rad_max": 3.1}]
# Configuration for the simulation
config = {"num_reads": 1000, "num_sweeps": 1000}
solve_func = neal.SimulatedAnnealingSampler().sample_qubo

# Initialize lists to store results
nodes_count = []
objective_no_process = []
objective_process = []
violations_count = []

# Loop over each parameter set
for params in params_list:
    data = SPData().gen_problem(**params) 
    data_copy = copy.deepcopy(data)
    
    # Solve without processing 
    qubo_model_bin_no_process = SPQuboBinary(data, process=False)
    answer_True_no_process = qubo_model_bin_no_process.solve(solve_func, **config)
    evaluation_no_process = SPEvaluation(data, answer_True_no_process['solution'])

    # Solve with processing
    qubo_model_bin_process = SPQuboBinary(data, process=True)
    answer_True_process = qubo_model_bin_process.solve(solve_func, **config)
    for key in qubo_model_bin_process.radar0:
        answer_True_process['solution'][key] = np.int8(0)
    for key in qubo_model_bin_process.radar1:
        answer_True_process['solution'][key] = np.int8(1)
    
    def convert_keys_to_strings(solution):  # To have some proper solution shape
        new_solution = {}
        for key, value in solution.items():
            if isinstance(key, tuple):
                key = 'x_' + '_'.join(map(str, key))
            new_solution[key] = value
        return new_solution
    answer_True_process['solution'] = convert_keys_to_strings(answer_True_process['solution'])
    evaluation_process = SPEvaluation(data, answer_True_process['solution'])

    # Store the number of nodes, N(x_min), and violations
    nodes_count.append(len(data_copy.G.nodes))  # Number of nodes in the graph
    objective_no_process.append(evaluation_no_process.get_objective())  # number of lidars activated in x_min without processing
    objective_process.append(evaluation_process.get_objective())  # number of lidars activated in x_min with processing

    # Count violated constraints
    violations = 0
    for constraint, violation_list in evaluation_process.check_solution().items():
        if len(violation_list) > 0:
            violations += len(violation_list)
    violations_count.append(violations)

norm = mcolors.Normalize(vmin=min(violations_count), vmax=max(violations_count))
cmap = plt.get_cmap('viridis')

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(nodes_count, objective_no_process, c=violations_count, cmap=cmap, norm=norm, marker='o', s=100, label='$M_1$')  # Increased size for 'o'
sc2 = ax.scatter(nodes_count, objective_process, c=violations_count, cmap=cmap, norm=norm, marker='x', s=150, label='$M_2$')  # Increased size for 'x'

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Number of Violated Constraints', fontsize=20)

ax.set_xlabel('Number N of Nodes', fontsize=26)
ax.set_ylabel('$||x_{min}||=\sum_{i=1}^{N}x_{i}$', fontsize=26)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(fontsize=20)
plt.grid(True)
plt.savefig("plot_results_2models_version1.png", dpi=300)
plt.show()
