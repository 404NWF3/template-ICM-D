# MCM/ICM Problem Types Reference

## Problem A: Continuous Optimization

**Characteristics:**
- Continuous decision variables
- Often involves differential equations, calculus of variations
- Objective: maximize/minimize continuous functions
- Constraints: inequalities, equalities, boundary conditions

**Common formulations:**
- Optimal control problems
- Resource allocation (continuous quantities)
- Shape optimization
- Trajectory planning

**Key techniques:**
- Gradient-based methods (steepest descent, Newton)
- Calculus of variations (Euler-Lagrange equations)
- Optimal control theory (Pontryagin's maximum principle)

**Example problems:**
- Designing optimal sailboat hull shape
- Planning spacecraft trajectory
- Optimizing drug dosage over time

---

## Problem B: Discrete/Combinatorial Optimization

**Characteristics:**
- Integer/binary decision variables
- NP-hard complexity
- Sequencing, scheduling, assignment decisions
- Exponential solution space

**Common formulations:**
- Traveling Salesman Problem (TSP)
- Vehicle Routing Problem (VRP)
- Knapsack problem
- Scheduling (job shop, flow shop)
- Facility location

**Key techniques:**
- Exact: branch-and-bound, cutting planes, dynamic programming
- Heuristic: genetic algorithms, simulated annealing, tabu search
- Problem-specific: Hungarian algorithm, Christofides algorithm

**Example problems:**
- Optimizing EV charging station placement
- Designing tournament schedule
- Assigning specialists to projects

---

## Problem C: Data-Driven Problems

**Characteristics:**
- Given dataset, build predictive/descriptive model
- Focus on accuracy, interpretability, generalization
- Machine learning, statistics, time series

**Common formulations:**
- Regression (predict continuous outcomes)
- Classification (predict categories)
- Clustering (discover groups)
- Time series forecasting

**Key techniques:**
- Regression: linear, polynomial, ridge/lasso
- Classification: logistic regression, decision trees, neural networks
- Clustering: k-means, hierarchical, DBSCAN
- Model selection: cross-validation, information criteria

**Example problems:**
- Predicting bee population decline
- Classifying network traffic patterns
- Modeling disease spread

---

## Problem D: Network/Graph Problems

**Characteristics:**
- Entities and relationships modeled as graph
- Nodes, edges, flows, paths
- Network topology matters

**Common formulations:**
- Shortest path (Dijkstra, Bellman-Ford)
- Maximum flow/minimum cut (Ford-Fulkerson)
- Minimum spanning tree (Kruskal, Prim)
- Facility location on networks
- Network design

**Key techniques:**
- Graph algorithms (NetworkX implementation)
- Minimum cost flow
- Steiner tree problems
- Centrality measures

**Example problems:**
- Optimizing road network after disaster
- Designing communication network
- Routing emergency vehicles

---

## Problem E: Environmental/Sustainability Systems

**Characteristics:**
- Multi-objective (environment + economic + social)
- Long time horizons
- Uncertainty in future conditions
- Policy relevance

**Common formulations:**
- Resource management (water, energy, fisheries)
- Climate impact assessment
- Land use planning
- Waste management optimization

**Key techniques:**
- Multi-objective optimization (Pareto fronts)
- Scenario analysis
- System dynamics modeling
- Life cycle assessment integration

**Example problems:**
- Optimizing food systems for sustainability
- Managing reservoir levels under drought
- Designing recycling programs

---

## Problem F: Policy/Operations Research

**Characteristics:**
- Multiple stakeholders with conflicting objectives
- Qualitative + quantitative factors
- Decision support rather than "optimal" solution
- Sensitivity to assumptions critical

**Common formulations:**
- Multi-criteria decision making (AHP, TOPSIS)
- Game theory (Nash equilibrium)
- Robust optimization
- Stochastic programming

**Key techniques:**
- Weighted sum methods
- Goal programming
- Scenario planning
- Monte Carlo simulation

**Example problems:**
- Evaluating climate policy options
- Designing fair electoral systems
- Allocating shared resources

---

## Problem Type Detection Flowchart

1. **Is data provided for model building?**
   - Yes → Likely Problem C (data-driven)
   - No → Continue

2. **Are decisions about connecting/configuring a network?**
   - Yes → Likely Problem D (network)
   - No → Continue

3. **Are decisions yes/no, sequencing, or assignment?**
   - Yes → Likely Problem B (discrete/combinatorial)
   - No → Continue

4. **Is problem about sustainability, environment, or long-term systems?**
   - Yes → Likely Problem E (sustainability)
   - No → Continue

5. **Are multiple stakeholders or policy choices involved?**
   - Yes → Likely Problem F (policy)
   - No → Likely Problem A (continuous)
