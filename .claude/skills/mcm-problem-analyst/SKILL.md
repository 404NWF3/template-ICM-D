---
name: mcm-problem-analyst
description: Extract and structure key optimization model elements from natural language MCM/ICM problem statements. Use when Claude needs to analyze mathematical modeling competition problems to identify: (1) Decision variables (continuous/discrete/integer), (2) Objective function (maximize/minimize/multi-objective), (3) Constraints (explicit and implicit like non-negativity, integrality), (4) Data requirements (given data vs parameters to find), (5) Solution accuracy requirements, or (6) Ambiguities requiring mathematical clarification
---

# MCM Problem Analyst

Extract optimization model components from problem statements in structured format.

## Analysis Workflow

Given a problem statement, systematically extract:

### 1. Decision Variables
- What quantities can we control?
- Variable type: continuous, discrete, binary, integer?
- Notation suggestion (e.g., x_i, y_j)

### 2. Objective Function
- Optimization direction: maximize or minimize?
- Single or multi-objective?
- Core objective (profit, cost, utility, distance, etc.)

### 3. Constraints
- **Explicit constraints**: stated directly in problem
- **Implicit constraints**: non-negativity, integrality, logical relationships
- Resource limitations, capacity bounds, temporal dependencies

### 4. Data Requirements
- What data is provided in problem?
- What parameters need external research/estimation?
- Data format and structure needs

### 5. Solution Accuracy
- Integer vs acceptable fractional solutions
- Precision requirements for continuous variables
- Approximation tolerance (if any)

## Ambiguity Resolution

Identify vague terms (e.g., "as much as possible", "reasonable", "efficient") and provide 3 mathematical interpretations:

**Example**: "maximize coverage"
1. Interpretation 1: Maximize number of served nodes
2. Interpretation 2: Minimize maximum distance to any node
3. Interpretation 3: Maximize total population within service radius

## Output Format

```markdown
## Decision Variables
- x₁: [description], [type]
- x₂: [description], [type]

## Objective Function
[Maximize/Minimize] [expression]

## Constraints
1. [Constraint 1]
2. [Constraint 2]
...

## Data Requirements
- Given: [list provided data]
- To find: [list required parameters]

## Ambiguities
- [Term]: 3 mathematical definitions
```

## Problem Type Detection

After extraction, identify problem category:
- **Problem A**: Continuous optimization (often involves differential equations, calculus of variations)
- **Problem B**: Discrete/combinatorial (integer programming, scheduling, routing)
- **Problem C**: Data-driven (regression, classification, time series)
- **Problem D**: Network/graph (shortest path, maximum flow, facility location)
- **Problem E**: Sustainable systems (resource allocation, energy optimization)
- **Problem F**: Policy (multi-criteria decision making)

See [references/problem-types.md](references/problem-types.md) for detailed characteristics of each MCM/ICM problem type.
