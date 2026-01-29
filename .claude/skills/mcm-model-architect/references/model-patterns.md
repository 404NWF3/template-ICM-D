# Classic Optimization Model Patterns

## Assignment Problems

**Problem:** Assign n workers to n tasks, minimize total cost.

**Formulation:**
```
Decision variables:
  x_ij ∈ {0,1}: 1 if worker i assigned to task j

Objective:
  min Σ_i Σ_j c_ij * x_ij

Constraints:
  Σ_j x_ij = 1  ∀i  (each worker one task)
  Σ_i x_ij = 1  ∀j  (each task one worker)
  x_ij ∈ {0,1}
```

**Solution:** Hungarian algorithm (O(n³)), or MIP solver.

---

## Knapsack Problem

**Problem:** Select items with maximum value subject to weight capacity.

**Formulation:**
```
Decision variables:
  x_i ∈ {0,1}: 1 if item i selected

Objective:
  max Σ_i v_i * x_i

Constraints:
  Σ_i w_i * x_i ≤ C  (capacity)
  x_i ∈ {0,1}
```

**Variations:**
- 0-1 knapsack (binary)
- Bounded knapsack (multiple copies)
- Unbounded knapsack (infinite copies)
- Multi-dimensional (multiple constraints)

---

## Facility Location Problem

**Problem:** Choose facility locations to minimize cost while serving demand.

**Formulation:**
```
Decision variables:
  x_ij ≥ 0: demand from j served by facility i
  y_i ∈ {0,1}: 1 if facility i opened

Objective:
  min Σ_i Σ_j c_ij * x_ij + Σ_i f_i * y_i

Constraints:
  Σ_i x_ij = d_j  ∀j  (demand satisfaction)
  Σ_j x_ij ≤ C_i * y_i  ∀i  (capacity)
  x_ij ≥ 0, y_i ∈ {0,1}
```

**Variations:**
- Uncapacitated (no C_i constraint)
- p-median (exactly p facilities)
- Covering (maximum distance constraint)

---

## Vehicle Routing Problem (VRP)

**Problem:** Route vehicles from depot to serve customers, minimize distance.

**Formulation:**
```
Decision variables:
  x_ijk ∈ {0,1}: 1 if vehicle k travels arc i→j
  u_i: auxiliary variable for subtour elimination

Objective:
  min Σ_k Σ_i Σ_j d_ij * x_ijk

Constraints:
  Σ_k Σ_j x_ijk = 1  ∀i≠depot  (each customer visited once)
  Σ_i x_ikh - Σ_j x_hjk = 0  ∀h,k  (flow conservation)
  u_i - u_j + n * x_ijk ≤ n-1  ∀i,j,k  (subtour elimination)
```

**Variations:**
- Capacitated VRP (vehicle capacity limits)
- VRP with time windows
- Split delivery VRP
- Periodic VRP

---

## Traveling Salesman Problem (TSP)

**Problem:** Visit all cities exactly once, return to start, minimize distance.

**Formulation (MTZ):**
```
Decision variables:
  x_ij ∈ {0,1}: 1 if travel from i to j
  u_i: order city i visited

Objective:
  min Σ_i Σ_j d_ij * x_ij

Constraints:
  Σ_j x_ij = 1  ∀i  (leave each city)
  Σ_i x_ij = 1  ∀j  (enter each city)
  u_i - u_j + n * x_ij ≤ n-1  ∀i≠j  (subtour elimination)
  x_ij ∈ {0,1}, u_i ∈ {1,2,...,n}
```

**Solution:** Exact (branch-and-cut) for n<100; heuristics for larger.

---

## Network Flow Problems

**Maximum Flow:**
```
Decision variables:
  f_ij: flow on edge i→j

Objective:
  max Σ_j f_source,j

Constraints:
  Σ_j f_ij - Σ_k f_ki = 0  ∀i≠source,sink  (flow conservation)
  0 ≤ f_ij ≤ capacity_ij
```

**Minimum Cost Flow:**
```
Decision variables:
  f_ij: flow on edge i→j

Objective:
  min Σ_i Σ_j cost_ij * f_ij

Constraints:
  Σ_j f_ij - Σ_k f_ki = b_i  ∀i  (supply/demand balance)
  0 ≤ f_ij ≤ capacity_ij
```

---

## Multi-Objective Optimization

**Problem:** Optimize multiple conflicting objectives simultaneously.

**Approaches:**

1. **Weighted Sum:**
   ```
   min w1*f1(x) + w2*f2(x) + ... + wk*fk(x)
   ```
   Vary weights to trace Pareto frontier.

2. **ε-Constraint:**
   ```
   min f1(x)
   s.t. f2(x) ≤ ε2
        f3(x) ≤ ε3
        ...
   ```
   Vary ε values to find Pareto points.

3. **Goal Programming:**
   ```
   min Σ_i (deviation from target_i)
   ```

**Pareto Optimality:** Solution x dominates y if x is better or equal on all objectives and strictly better on at least one.

---

## Stochastic Programming

**Problem:** Optimize under uncertainty (random parameters).

**Two-Stage Formulation:**
```
Here-and-now variables: x (decided before uncertainty reveals)
Wait-and-see variables: y_s (decided after scenario s occurs)

Objective:
  min c'x + Σ_s p_s * E[Q(x,s)]

where Q(x,s) is recourse cost for scenario s.
```

**Expected Value of Perfect Information (EVPI):**
```
EVPI = WS - EEV
```
Where WS = wait-and-see (know future), EEV = expected value of here-and-now solution.

---

## Dynamic Programming

**Structure:** Optimal substructure + overlapping subproblems.

**Bellman Equation:**
```
V(s) = max_a [R(s,a) + γ * Σ_s' P(s'|s,a) * V(s')]
```

**Applications:**
- Inventory management
- Resource allocation over time
- Sequential decision making

**Curse of Dimensionality:** State space grows exponentially with variables.

---

## Model Selection Heuristics

| Model Type | Use When | Complexity |
|------------|----------|------------|
| LP | Continuous, linear objective/constraints | P (easy) |
| MIP | Integer decisions, linear constraints | NP-hard |
| NLP | Smooth nonlinearities | NP-hard |
| DP | Sequential decisions, optimal substructure | P if small state space |
| Network | Graph structure, flows | P (easy) |
| Metaheuristic | NP-hard, large instances, acceptable approx | Fast but no guarantees |
