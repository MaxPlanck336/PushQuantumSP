from data.sp_data import SPData
from models import SPCplex
from evaluation.evaluation import SPEvaluation
from plotting.sp_plot import SPPlot
import neal
import numpy as np  
from models import SPQuboBinary
import networkx as nx
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

params = {"lidar_density": 0.6, "street_point_density": 0.6}
data = SPData().create_problem_from_glb_file(**params)
data_copy = copy.deepcopy(data)
plt = SPPlot(data).plot_problem()
plt.show()

cplex_model = SPCplex(data)
config = {"num_reads":1000,"num_sweeps":1000}
solve_func = neal.SimulatedAnnealingSampler().sample_qubo

# Solution with process=False or True
qubo_model_bin_process = SPQuboBinary(data, process=True) #choose whether to process or not
print("Shape of the QUBO matrix with process=False:", qubo_model_bin_process.model.shape)

answer = qubo_model_bin_process.solve(solve_func, **config)
for key in qubo_model_bin_process.radar0:
        answer['solution'][key] = np.int8(0)
for key in qubo_model_bin_process.radar1:
        answer['solution'][key] = np.int8(1)
    
def convert_keys_to_strings(solution):
    new_solution = {}
    for key, value in solution.items():
        if isinstance(key, tuple):
            key = 'x_' + '_'.join(map(str, key))
        new_solution[key] = value
    return new_solution
answer['solution'] = convert_keys_to_strings(answer['solution'])
evaluation_process = SPEvaluation(data, answer['solution'])

evaluation = SPEvaluation(data, answer["solution"])

print(f"objective = {evaluation.get_objective()}")
for constraint, violations in evaluation.check_solution().items():
    if len(violations) > 0:
        print(f"constraint {constraint} was violated {len(violations)} times")

plt = SPPlot(data_copy, evaluation).plot_solution(hide_never_covered = True)
plt.show()
