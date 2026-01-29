---
name: mcm-solver-engineer
description: Generate production-ready Python code for solving MCM/ICM optimization models. Use when Claude needs to: (1) Create complete Python code using scipy.optimize, cvxpy, or COPT libraries, (2) Implement data preprocessing (missing value imputation, normalization), (3) Add feasibility verification (check constraint satisfaction), (4) Output objective function value and optimal variable values, or (5) Generate heuristic algorithm frameworks (Genetic Algorithm, Simulated Annealing) for NP-hard combinatorial problems
---

# MCM Solver Engineer

Generate executable Python optimization code with data handling and validation.

## Code Structure Template

Every solver script should follow this structure:

```python
# 1. Imports and setup
# 2. Data loading and preprocessing
# 3. Model formulation
# 4. Solver configuration
# 5. Solution extraction
# 6. Feasibility verification
# 7. Results output
```

## Library Selection Guide

### For LP/MIP Problems
```python
# Recommended: scipy.optimize.linprog (built-in, no install)
from scipy.optimize import linprog

# Alternative: PuLP (more intuitive API)
import pulp

# Large-scale: COPT (fastest for academic use)
import copt
```

### For NLP Problems
```python
# Recommended: scipy.optimize.minimize
from scipy.optimize import minimize

# Advanced: cvxpy (convex problems)
import cvxpy as cp
```

### For Graph/Network Problems
```python
import networkx as nx
```

## Data Preprocessing Pattern

```python
def preprocess_data(raw_data):
    """Handle missing values, normalize, validate types."""
    # 1. Handle missing values
    # 2. Normalize if needed (0-1 scaling)
    # 3. Type conversion
    # 4. Range validation
    return processed_data
```

## Feasibility Verification

**CRITICAL**: Always verify solution satisfies all constraints:

```python
def verify_feasibility(solution, constraints, tolerance=1e-6):
    """Check all constraints are satisfied."""
    violations = []
    for name, (check_func, params) in constraints.items():
        if not check_func(solution, **params):
            violations.append(name)
    return violations
```

## Solver Code Patterns

### Linear Programming (scipy.optimize)

```python
from scipy.optimize import linprog
import numpy as np

# Minimize: c^T x
# Subject to: A_ub x <= b_ub, A_eq x = b_eq, bounds

def solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        return {
            'status': 'optimal',
            'objective': result.fun,
            'variables': result.x,
            'message': result.message
        }
    else:
        return {'status': 'failed', 'message': result.message}
```

### Integer Programming (PuLP)

```python
import pulp

def solve_mip(objective_func, constraints, var_types, bounds):
    prob = pulp.LpProblem("MCM_Problem", pulp.LpMinimize)

    # Create variables
    x = [pulp.LpVariable(f'x{i}', lowBound=bounds[i][0],
                         upBound=bounds[i][1], cat=var_types[i])
         for i in range(len(var_types))]

    # Add objective
    prob += objective_func(x)

    # Add constraints
    for constr in constraints:
        prob += constr['lhs'](x) <= constr['rhs']

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        'status': pulp.LpStatus[prob.status],
        'objective': pulp.value(prob.objective),
        'variables': [v.value() for v in x]
    }
```

### Genetic Algorithm Template

See [scripts/genetic_algorithm.py](scripts/genetic_algorithm.py) for production-ready GA framework with:
- Population initialization
- Fitness evaluation
- Selection (tournament/roulette)
- Crossover operators (single-point, uniform, order-based)
- Mutation operators (swap, scramble, inversion)
- Elitism preservation
- Convergence detection

### Simulated Annealing Template

See [scripts/simulated_annealing.py](scripts/simulated_annealing.py) for production-ready SA framework with:
- Initial solution generation
- Temperature scheduling (exponential, logarithmic)
- Neighbor generation
- Acceptance probability (Metropolis criterion)
- Cooling schedule
- Reheating option

## Output Formatting

```python
def format_results(solution, objective, solve_time, verification):
    """Generate clean, interpretable output."""
    print(f"{'='*50}")
    print(f"Optimization Result")
    print(f"{'='*50}")
    print(f"Status: {solution['status']}")
    print(f"Objective Value: {objective:.4f}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"\nOptimal Variables:")
    for i, val in enumerate(solution['variables']):
        print(f"  x[{i}] = {val:.4f}")
    print(f"\nConstraint Verification:")
    if verification['violations']:
        print(f"  VIOLATIONS: {verification['violations']}")
    else:
        print(f"  All constraints satisfied ✓")
    return {
        'solution': solution,
        'objective': objective,
        'verification': verification
    }
```

## Debug Checklist

When solver fails:
1. Check constraint consistency (no contradictions)
2. Verify bounded feasible region exists
3. Check numerical scaling (avoid very large/small numbers)
4. Validate data types (float vs int)
5. Test with simplified instance

## Problem-Specific Solvers

### For Problem D (Network): Use NetworkX
- `nx.shortest_path_length()` for distances
- `nx.maximum_flow()` for flow problems
- `nx.minimum_spanning_tree()` for connectivity

### For Problem B (Discrete): Use heuristics
- GA for permutation problems (TSP, scheduling)
- SA for configuration problems
- OR-Tools for VRP variants

### For Problem A (Continuous): Use scipy.optimize
- `minimize()` with appropriate method
- `linprog()` for linear constraints
- `fsolve()` for systems of equations
