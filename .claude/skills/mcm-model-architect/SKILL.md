---
name: mcm-model-architect
description: Design and compare mathematical optimization models for MCM/ICM problems. Use when Claude needs to: (1) Formulate optimization models with objective functions and constraints, (2) Generate 3-5 solution approaches (linear programming, integer programming, heuristics), (3) Create comparison tables by difficulty, data needs, accuracy, and applicability, (4) Recommend approaches suitable for 4-day competition timeframe, or (5) Provide model simplification strategies (linearization, dimensionality reduction) with error analysis
---

# MCM Model Architect

Design optimization models and select appropriate solution strategies.

## Model Formulation

Based on extracted problem elements, formulate mathematical model:

### Standard Form

**For minimization:**
```
minimize: f(x)
subject to:
    g_i(x) ≤ 0,  i = 1, ..., m
    h_j(x) = 0,  j = 1, ..., p
    x ∈ X
```

**For maximization:** convert to minimization via `max f(x) = min -f(x)`

### Notation Standards
- Use LaTeX for all mathematical expressions
- Define all symbols clearly
- Specify variable domains (R⁺, Z⁺, {0,1})
- Index sets: i ∈ I, j ∈ J, etc.

## Solution Approach Generation

Generate 3-5 approaches with this classification:

### Exact Methods
- **Linear Programming (LP)**: Continuous variables, linear objective/constraints
- **Integer Programming (IP/MIP)**: Integer/binary variables, linear constraints
- **Nonlinear Programming (NLP)**: Smooth nonlinearities
- **Dynamic Programming (DP)**: Sequential decision making, optimal substructure

### Approximation/Heuristic Methods
- **Genetic Algorithm (GA)**: Population-based, crossover/mutation
- **Simulated Annealing (SA)**: Temperature-based acceptance
- **Particle Swarm Optimization (PSO)**: Swarm intelligence
- **Ant Colony Optimization (ACO)**: Pheromone-based pathfinding
- **Tabu Search**: Memory-based local search

### Problem-Specific Methods
- **Network algorithms**: Dijkstra, Ford-Fulkerson, minimum spanning tree
- **Assignment algorithms**: Hungarian algorithm
- **Knapsack algorithms**: Branch-and-bound, pseudo-polynomial

## Comparison Matrix

| Approach | Difficulty | Data Needs | Accuracy | Speed | Best For |
|----------|-----------|------------|----------|-------|----------|
| Method 1 | ... | ... | ... | ... | ... |
| Method 2 | ... | ... | ... | ... | ... |

**Rating scale:**
- Difficulty: Low/Medium/High
- Data Needs: Minimal/Moderate/Extensive
- Accuracy: Exact/High/Medium/Low
- Speed: Fast/Medium/Slow

## Recommendation Criteria

For 4-day competition, prioritize:
1. **Implementability**: Can you code it in Python/MATLAB in 1 day?
2. **Interpretability**: Can you explain and defend results?
3. **Robustness**: Does it handle edge cases?
4. **Result quality**: Good enough for competition success?

**Default recommendation**: Start with exact method (if tractable), fall back to heuristic if needed.

## Model Simplification

When model is too complex:

### Linearization Techniques
- **Absolute value**: |x| ≤ t → x ≤ t, -x ≤ t
- **Min/max**: min(x,y) ≤ z → x ≤ z, y ≤ z, z ≤ x + M·b, z ≤ y + M·(1-b)
- **Products**: x·y (binary × continuous) → use big-M

### Dimensionality Reduction
- **Aggregation**: Combine similar variables
- **Time discretization**: Continuous → discrete time periods
- **Spatial clustering**: Group nearby locations

### Relaxation Strategies
- **LP relaxation**: Solve continuous version, round intelligently
- **Lagrangian relaxation**: Move difficult constraints to objective

**Error quantification**: Always estimate bound from simplification.

## Problem-Specific Guidance

See [references/model-patterns.md](references/model-patterns.md) for:
- Classic formulations: TSP, VRP, knapsack, facility location
- Network flow models
- Multi-objective formulations
- Stochastic programming basics

## Technical Stack Recommendations

| Solver | Use Case | Python Package | License |
|--------|----------|---------------|---------|
| scipy.optimize | Small-medium LP, NLP | Built-in | Free |
| PuLP | LP/MIP modeling | `pip install pulp` | Free |
| COPT | Large-scale LP/MIP | `pip install copt` | Free academic |
| Gurobi | Commercial solver | `pip install gurobipy` | Academic free |
| CPLEX | Commercial solver | `pip install cplex` | Academic free |
| OR-Tools | Constraint programming, routing | `pip install ortools` | Free |
| DEAP | Genetic algorithms | `pip install deap` | Free |
| NetworkX | Graph algorithms | `pip install networkx` | Free |
