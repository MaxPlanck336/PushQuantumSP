import neal
import numpy as np  
from data.sp_data import SPData
from models import SPQuboBinary
from evaluation.evaluation import SPEvaluation
from plotting.sp_plot import SPPlot
import networkx as nx
import copy


params =    {"version": 3, "num_cols": 43, "rad_max": 2.8}
#{"version": 3, "num_cols": 4, "rad_max": 3.2}
data = SPData().gen_problem(**params) 
data_copy = copy.deepcopy(data)
plt = SPPlot(data).plot_problem()
plt.show()

config = {"num_reads":1000,"num_sweeps":1000}
solve_func = neal.SimulatedAnnealingSampler().sample_qubo

"""
We are comparing the initial SPQuboBinary implemented (when process=False) 
with the new SPQuboBinary implemented (when process=True)
"""

# Solution Without any data porcessing (matrix reduction)
qubo_model_bin_no_process = SPQuboBinary(data, process=False)
print("Shape of the QUBO matrix with process=False:", qubo_model_bin_no_process.model.shape)

answer_True_no_process = qubo_model_bin_no_process.solve(solve_func, **config)
print(answer_True_no_process['solution'])
evaluation_no_process = SPEvaluation(data, answer_True_no_process['solution'])
print(f"objective with process=False = {evaluation_no_process.get_objective()}")
print('computation time with process=False: ', answer_True_no_process['runtime'])
for constraint, violations in evaluation_no_process.check_solution().items():
    if len(violations) > 0:
        print(f"constraint {constraint} was violated {len(violations)} times")

plt = SPPlot(data_copy, evaluation_no_process).plot_solution(hide_never_covered=True)
plt.show()

# Solution with data processing 
qubo_model_bin_process = SPQuboBinary(data, process=True)


print("Shape of the QUBO matrix with process=True:", qubo_model_bin_process.model.shape)

answer_True_process = qubo_model_bin_process.solve(solve_func, **config) #This is the solution for the reduced Q
for key in qubo_model_bin_process.radar0: #We are readding the trivial terms 
    answer_True_process['solution'][key] = np.int8(0) 
for key in qubo_model_bin_process.radar1:
    answer_True_process['solution'][key] = np.int8(1) #We are readding the trivial terms


def convert_keys_to_strings(solution): # To have some proper solution shape
    new_solution = {}
    for key, value in solution.items():
        if isinstance(key, tuple):  # Vérifie si la clé est un tuple
            # Convertit le tuple en une chaîne de caractères (par exemple, 'x_0.0_0.0_2.5_0_-10')
            key = 'x_' + '_'.join(map(str, key))
        new_solution[key] = value
    return new_solution
answer_True_process['solution'] = solution_process_str_keys = convert_keys_to_strings(answer_True_process['solution'])
evaluation_process = SPEvaluation(data, answer_True_process['solution'])


print('computation time with process=True: ', answer_True_process['runtime']) 

print(f"objective with process=True = {evaluation_process.get_objective()}") #Print the minimal number of Lidar 
for constraint, violations in evaluation_process.check_solution().items():
    if len(violations) > 0:
        print(f"constraint {constraint} was violated {len(violations)} times") #Print the conditions that were violated

plt = SPPlot(data_copy, evaluation_process).plot_solution(hide_never_covered=True)
plt.show()

# Compare the two answers

# Compare the two solutions and print where the differences are
if np.array_equal(evaluation_no_process.solution, evaluation_process.solution):
    print("The solutions are identical.")

else:
    print("The solutions are different.")

    # Find where the values differ
    differences = np.where(sevaluation_no_process.solution != evaluation_process.solution)
    
    # Print the indices where differences occur
    print(f"Differences found at indices: {differences}")
    
