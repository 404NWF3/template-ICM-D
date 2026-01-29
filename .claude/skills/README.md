# MCM/ICM Mathematical Modeling Skills

A comprehensive set of Claude Code skills for solving MCM/ICM mathematical modeling competition problems. These skills implement the "AI → Human Verification → Optimization" workflow with proper AI usage attribution.

## Skills Overview

### 1. mcm-problem-analyst
**Agent A: Problem Analyst**

Extracts optimization model components from natural language problem statements:
- Decision variables (continuous/discrete/integer)
- Objective function (maximize/minimize/multi-objective)
- Constraints (explicit and implicit)
- Data requirements
- Ambiguity resolution with 3 mathematical interpretations

**Use when:** You have a problem statement and need to identify the mathematical model structure.

**Reference:** [references/problem-types.md](mcm-problem-analyst/references/problem-types.md) - Detailed characteristics of all MCM/ICM problem types (A-F)

---

### 2. mcm-model-architect
**Agent B: Model Architect**

Designs optimization models and selects solution strategies:
- Mathematical model formulation (LaTeX format)
- 3-5 solution approach generation with comparison tables
- Technical stack recommendations (scipy, COPT, Gurobi, etc.)
- Model simplification strategies (linearization, dimensionality reduction)

**Use when:** You need to formulate the mathematical model and decide which algorithm to use.

**Reference:** [references/model-patterns.md](mcm-model-architect/references/model-patterns.md) - Classic optimization formulations (TSP, VRP, knapsack, facility location, etc.)

---

### 3. mcm-solver-engineer
**Agent C: Solver Engineer**

Generates production-ready Python optimization code:
- Complete solver implementations (scipy.optimize, cvxpy, COPT)
- Data preprocessing and validation
- Feasibility verification (constraint checking)
- Heuristic algorithm frameworks (Genetic Algorithm, Simulated Annealing)

**Use when:** You need executable Python code to solve your optimization model.

**Scripts:**
- [scripts/genetic_algorithm.py](mcm-solver-engineer/scripts/genetic_algorithm.py) - Production-ready GA with multiple operators
- [scripts/simulated_annealing.py](mcm-solver-engineer/scripts/simulated_annealing.py) - Production-ready SA with cooling schedules

---

### 4. mcm-validation-analyst
**Agent D: Validation Analyst**

Performs sensitivity analysis and result interpretation:
- One-way and two-way sensitivity analysis design
- Visualization generation (line plots, heatmaps, tornado diagrams)
- Robustness assessment and enhancement strategies
- Practical solution interpretation

**Use when:** You need to validate model reliability and analyze parameter sensitivity.

**Script:** [scripts/sensitivity_analysis.py](mcm-validation-analyst/scripts/sensitivity_analysis.py) - Complete sensitivity analysis toolkit

---

### 5. mcm-academic-writer
**Agent E: Academic Writer**

Transforms models and results into competition-quality academic writing:
- Model Establishment section with LaTeX formatting
- Three-paragraph abstract structure (background → method/results → sensitivity/conclusions)
- Mathematical notation and derivation explanations
- LaTeX quick reference

**Use when:** You need to write the final paper with proper academic formatting.

**Asset:** [assets/latex-template.tex](mcm-academic-writer/assets/latex-template.tex) - Complete LaTeX paper template

---

## Typical Workflow

```
Problem Statement
        ↓
    [mcm-problem-analyst]
    Extract: variables, objectives, constraints
        ↓
    [mcm-model-architect]
    Formulate model + select algorithm
        ↓
    [mcm-solver-engineer]
    Generate Python solver code
        ↓
    [mcm-validation-analyst]
    Sensitivity analysis + validation
        ↓
    [mcm-academic-writer]
    Write paper sections + abstract
```

## Installation

Each skill is a self-contained package. To use in Claude Code:

1. Copy skill directories to your Claude skills folder
2. Skills will auto-trigger based on their descriptions
3. Or invoke manually with `/skill-name`

## Important Notes

### Human-in-the-Loop Verification
- **Always verify AI-generated code** before submitting
- Check that solutions satisfy all constraints
- Validate objective function direction (max vs min)
- Ensure inequality signs are correct

### AI Usage Attribution (2026 MCM Requirements)
Document AI usage in your paper appendix:
```
We used AI (Claude Code) to assist with:
- Initial problem analysis and variable identification
- Python code generation for optimization solvers
- Sensitivity analysis visualization code
- Academic writing suggestions

All AI-generated content was reviewed, verified, and modified
by team members before inclusion in the final paper.
```

### Compliance
- AI content must be reviewed and modified by humans
- Cannot directly copy-paste as final submission
- Must declare AI usage scope in appendix

## Skill Structure

Each skill follows Claude Code standards:
```
skill-name/
├── SKILL.md (required - metadata + instructions)
└── [optional resources]
    ├── scripts/ (executable code)
    ├── references/ (documentation for context)
    └── assets/ (files for output)
```

## Quick Reference

| Problem Type | Recommended Approach |
|--------------|---------------------|
| Problem A (continuous) | scipy.optimize.minimize, gradient-based methods |
| Problem B (discrete) | GA/SA heuristics, MIP solvers |
| Problem C (data-driven) | scikit-learn, regression/classification |
| Problem D (network) | NetworkX, min-cost flow algorithms |
| Problem E (sustainability) | Multi-objective optimization, Pareto fronts |
| Problem F (policy) | Multi-criteria decision making, robust optimization |

## License

These skills are designed for educational use in mathematical modeling competitions.
