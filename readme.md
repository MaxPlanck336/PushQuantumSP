# PUSH Quantum Hackathon 2024 -- Sensor Positioning (AQORA)

## Example_qubo_generator.py

This script showcases a naive approach versus a custom preprocessing approach to solving a sensor placement optimization problem using QUBO (Quadratic Unconstrained Binary Optimization). It’s designed to highlight how preprocessing can enhance solution quality and computational efficiency.

### Key Goals:

1.	Demonstrate problem-solving efficiency using preprocessing.
2.	Compare outcomes from naive and custom approaches.
3.	Provide visual and metric-based analysis to justify improvements.

### Workflow Overview

1.	Problem Setup:
   - Generate sensor placement data (SPData) with customizable parameters.
   - Visualize the problem’s layout for clarity.
2.	Approach 1: Naive Solution:
   - Directly solves the full QUBO matrix without preprocessing.
   - Measures runtime, objective value, and constraint satisfaction.
3.	Approach 2: Preprocessed Solution:
   - Reduces the QUBO matrix via preprocessing.
   - Solves the simplified problem, reintegrates terms, and evaluates results.
4.	Comparison:
   - Highlights differences in runtime, solution quality, and constraints between the two methods.

### Key Features for Hackathon Use

1.	Plug-and-Play Problem Generation:
Modify parameters in params to quickly generate new sensor placement problems:
```python
params = {"version": 3, "num_cols": 43, "rad_max": 2.8}
```

2.	Solver Configuration:
Adjust solver parameters for optimal performance:
```python
config = {"num_reads": 1000, "num_sweeps": 1000}
```

3.	Preprocessing Advantage:
The custom preprocessing approach significantly reduces problem size, offering:
   - Faster computation.
   - Improved scalability for larger datasets.
4.	Visual Feedback:
Plots provide immediate insights into:
   - Problem layout.
   - Differences between solutions.
5.	Metrics for Evaluation:
   - Objective value: Measures the number of Lidars required.
   - Runtime: Benchmarks computational efficiency.
   - Constraint satisfaction: Validates solution feasibility.

### How to Run

1.	Install Required Libraries:
```python
pip install neal numpy networkx matplotlib
```
Ensure custom modules (SPData, SPQuboBinary, SPEvaluation, SPPlot) are available.

3.	Run the Script:
```python
python <script_name>.py
```

3.	Analyze Results:
- Compare terminal outputs for runtimes and metrics.
- Review plots for solution differences.

### Summary of Results (Sample Output)

Naive Approach:
- Runtime: 4.9 seconds

Preprocessed Approach:
- Runtime: 4.0 seconds

Time Saved: 19.56 %

Preprocessing saves time and enhances results!

## Example_cplex_generator.py

This script tackles a real-world sensor placement problem, optimizing coverage in urban environments while minimizing resources like Lidars. It compares solutions from QUBO-based Simulated Annealing and CPLEX, providing actionable insights and impactful visualizations for practical applications such as smart city planning.

### Workflow

1.	Problem Setup:
   - Generate a realistic urban sensor placement problem using .glb data.
   - Visualize the problem layout.
2.	QUBO Solution:
   - Formulate and solve the problem with or without preprocessing (process=False/True) using Simulated Annealing (neal).
3.	CPLEX Solution:
   - Solve the same problem using precise optimization for comparison.
4.	Evaluation:
   - Compute objective value (minimum sensors).
   - Validate constraint satisfaction.
   - Visualize solutions for decision-making.

### Key Features

- Real-world relevance: Simulates urban planning scenarios.
- Solver flexibility: Compare fast heuristic (QUBO) with precise (CPLEX) methods.
- Metrics and visuals: Outputs objective values, constraint checks, and clear plots for coverage.

### How to Use

1.	Install dependencies:
```python   
pip install neal numpy matplotlib
```
2.	Run the script:
```python   
python <script_name>.py
```
3.	Analyze results:

- Compare terminal metrics and constraint violations.
- Use plots for visual insights.

### Summary

Naive Approach:
-	Runtime: 0.0011 seconds
-	Objective Value: 10

Preprocessed Approach:
-	Runtime: 0.0006 seconds
-	Objective Value: 6

Time Saved: 45 %

Leverage this script to tackle real-world optimization problems and present actionable results effectively!

## Benchmark.py
### List of issues in classical simulation algorithms :

Unavailable solver :
- ("QAOA", "ibm"): Access to IBM is currently not available on Luna.
- ("VQE", "ibm"): Access to IBM is currently not available on Luna.
- ("BF", ): Error message: Solver unavailable
- ("SAGA_PL","dwave"): Error message: Solver unavailable
- ("SAGA_PW","dwave"): Error message: Solver unavailable

Solver with docs issue :
- ("PT", "dwave"): invalid param (rtol)
- ("QLSA", "dwave"): invalid param (rtol)
- ("TS","dwave"): invalid params (num_reads, tenure, timeout, initial_state_generator)
- ("DS","dwave"): invalid param (rtol)
- ("QLTS","dwave"): invalid param (rtol)
