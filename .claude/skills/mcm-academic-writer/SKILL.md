---
name: mcm-academic-writer
description: Transform MCM/ICM models and results into rigorous academic writing. Use when Claude needs to: (1) Write the "Model Establishment" section with clear mathematical formulations and LaTeX notation, (2) Generate competition abstracts using the three-paragraph structure (background+method, algorithm+results, sensitivity+advantages), (3) Create academic English text with proper terminology and logical flow, or (4) Format mathematical content for LaTeX/Word with symbol definitions and derivations
---

# MCM Academic Writer

Transform technical work into publication-quality academic text for MCM/ICM competitions.

## Writing Principles

- **Active voice preferred**: "We formulate" not "It is formulated"
- **Present tense for model**: "Our model optimizes"
- **Past tense for experiments**: "We tested", "The algorithm converged"
- **Concrete specificity**: Include actual numbers, not "significant improvement"
- **Logical flow**: Each sentence justifies the next

## Model Establishment Section

### Section Structure

```markdown
## Model Establishment

### 5.1 Problem Analysis and Assumptions
[Brief context, key simplifying assumptions]

### 5.2 Notation
[Symbol table with definitions]

### 5.3 Model Formulation
[Objective function, constraints, complete formulation]

### 5.4 Solution Approach
[Algorithm choice justification]
```

### Notation Table Template

```latex
\\begin{table}[h]
\\centering
\\begin{tabular}{ll}
\\hline
Symbol & Description \\\\
\\hline
$x_{ij}$ & Decision variable: amount shipped from i to j \\\\
$c_{ij}$ & Unit transportation cost from i to j \\\\
$d_j$ & Demand at location j \\\\
$s_i$ & Supply capacity at source i \\\\
\\hline
\\end{tabular}
\\caption{Model Notation}
\\end{table}
```

### Mathematical Formulation Pattern

```markdown
### 5.3 Mathematical Model

We formulate the problem as a **mixed-integer linear program (MILP)**.

**Decision Variables:**
- $x_{ij} \in \mathbb{R}_{\ge 0}$: Quantity shipped from facility $i$ to customer $j$
- $y_i \in \{0,1\}$: Binary variable indicating whether facility $i$ is opened

**Objective Function:**
Minimize total cost:
$$
\\min \\sum_{i \\in I} \\sum_{j \\in J} c_{ij} x_{ij} + \\sum_{i \\in I} f_i y_i
$$

**Constraints:**

1. **Demand satisfaction**: Each customer must receive full demand
$$
\\sum_{i \\in I} x_{ij} = d_j, \\quad \\forall j \\in J
$$

2. **Capacity limitation**: Shipments cannot exceed facility capacity
$$
\\sum_{j \\in J} x_{ij} \\le C_i y_i, \\quad \\forall i \\in I
$$

3. **Non-negativity and binary**
$$
x_{ij} \\ge 0, \\quad y_i \\in \\{0,1\\}, \\quad \\forall i \\in I, j \\in J
$$

**Model size:** $|I| \\cdot |J|$ continuous variables, $|I|$ binary variables, $|I| + |J|$ constraints.
```

### Derivation Logic

When deriving constraints, explain rationale:

> "Constraint (2) ensures that shipments from a facility cannot exceed its capacity $C_i$. The binary variable $y_i$ acts as a switch: when $y_i = 0$, the facility is closed and capacity becomes zero; when $y_i = 1$, the facility operates at full capacity."

## Abstract Writing

### Three-Paragraph Structure

**Paragraph 1: Background and Model**
- Hook: Why problem matters
- Approach: "[Specific problem type] modeled as [model class]"
- Scope: What aspects are considered

**Paragraph 2: Method and Results**
- Algorithm: "[Algorithm name] with [key innovation]"
- Results: SPECIFIC NUMBERS required
- Comparison: "Improved X by Y% over baseline"

**Paragraph 3: Validation and Conclusions**
- Sensitivity: Key parameter insights
- Advantages: Model strengths
- Impact: Practical implications

### Abstract Template

```markdown
[Problem Name] presents a critical challenge in [field] due to [key difficulty].
We develop a [model class] that [key modeling innovation]. Our approach integrates
[main techniques] to address [core problem aspect].

We employ [algorithm name] enhanced with [specific technique] to solve the model.
On [test instance], our method achieves [specific metric] of [value], representing
a [percentage]% improvement over [baseline method]. The optimal solution reduces
[objective] by [amount] while satisfying all [key constraint type] requirements.

Sensitivity analysis reveals [key finding: parameter X has Y impact on Z].
Our model demonstrates [strength 1], [strength 2], and [strength 3], making it
suitable for [application area]. This work provides [practical contribution]
for [stakeholders].
```

### Example: Transportation Problem Abstract

```markdown
Urban logistics optimization presents critical challenges due to rising congestion
and environmental concerns. We develop a multi-objective vehicle routing model
that simultaneously minimizes delivery cost, time, and carbon emissions. Our
approach integrates time-window constraints, capacity limitations, and traffic
patterns.

We employ a non-dominated sorting genetic algorithm (NSGA-II) enhanced with
adaptive crossover operators to solve the bi-objective model. On a dataset of
150 customers across 5 districts, our method identifies 12 Pareto-optimal
solutions, with the best compromise solution reducing total distance by 18.3%
and emissions by 22.7% compared to the baseline greedy approach. The optimal
routing serves all customers within specified time windows using 23% fewer
vehicles.

Sensitivity analysis reveals that fuel price fluctuations of ±30% change total
cost by only ±8%, demonstrating robust solution stability. Our model shows
strong scalability, handling up to 500 customers with 2.3-minute solve times,
making it practical for both daily operations and strategic planning. This
work provides logistics managers with a flexible tool for sustainable,
cost-effective delivery optimization.
```

## LaTeX Quick Reference

### Common Mathematical Expressions

```latex
% Sets
$\mathbb{R}$ (real numbers), $\mathbb{Z}$ (integers), $\mathbb{N}$ (natural)

% Subsets and ranges
$\forall i \in I$ (for all), $\exists j \in J$ (there exists)
$x_{ij} \in \{0,1\}$ (binary), $x \ge 0$ (non-negative)

% Sums and products
$\sum_{i=1}^{n} x_i$ (sum), $\prod_{j=1}^{m} y_j$ (product)

% Optimization
$\min f(x)$, $\max g(x)$
subject to: $\text{s.t.}$

% Greek letters
$\alpha, \beta, \gamma, \lambda, \mu, \sigma, \epsilon$

% Relations
$\le$ (leq), $\ge$ (geq), $\neq$ (neq), $\approx$ (approx)
```

### Algorithm Pseudocode

```latex
\\begin{algorithm}
\\caption{Genetic Algorithm for VRP}
\\begin{algorithmic}[1]
\\STATE Initialize population $P$ of size $N$
\\WHILE{not converged}
    \\STATE Evaluate fitness for each chromosome
    \\STATE Select parents via tournament selection
    \\STATE Apply crossover with probability $p_c$
    \\STATE Apply mutation with probability $p_m$
    \\STATE Replace worst solutions with offspring
\\ENDWHILE
\\RETURN Best solution found
\\end{algorithmic}
\\end{algorithm}
```

## Quality Checklist

Before submitting:

- [ ] All symbols defined in notation table
- [ ] Every constraint explained in plain English
- [ ] Objective function direction clearly stated (min/max)
- [ ] Algorithm choice justified
- [ ] Specific numerical results included
- [ ] Comparison to baseline provided
- [ ] Sensitivity findings summarized
- [ ] No vague terms like "significant" without numbers
- [ ] LaTeX syntax verified
- [ ] Word count within limits (abstract: 1 page max)

See [assets/latex-template.tex](assets/latex-template.tex) for complete paper template.
