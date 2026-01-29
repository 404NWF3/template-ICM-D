# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses `uv` for Python package management (not pip/conda). Python 3.10.19 is specified in `.python-version`.

### Initial Setup

```bash
# Install dependencies via uv
uv sync

# Or install with dev dependencies
uv sync --all-extras
```

### Running Python Code

```bash
# Run a Python script
uv run python script.py

# Run the main entry point (tests all imports)
uv run python main.py

# Start Jupyter Lab
uv run jupyter lab
```

## Project Architecture

This is a **template repository** for Operations Research and Optimization problems. The project structure is intentionally minimal:

- `main.py` - Simple import test script that validates all OR/Optimization libraries are installed
- `pyproject.toml` - UV project configuration with all OR/Optimization dependencies
- `README.md` - Basic project documentation (mentions `notebooks/`, `data/`, `scripts/` directories that should be created as needed)

## Key Dependencies

The project includes a comprehensive OR/Optimization toolkit:

| Category | Libraries |
|----------|-----------|
| **Data manipulation** | pandas, polars, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Graph/network** | networkx, igraph, pyvis |
| **Linear/Mixed-Integer Programming** | ortools, pulp, gurobipy |
| **Metaheuristics** | scikit-opt (sko), pygad |
| **Simulation** | simpy |
| **Decision Analysis/MCDM** | ahp-topsis, pydecision, SALib (sensitivity analysis) |
| **General optimization** | scipy |

## Working with Notebooks

The project is designed for Jupyter-based exploration:

```bash
# Start Jupyter Lab
uv run jupyter lab
```

When creating notebooks, organize them in appropriate directories:
- `notebooks/` - Jupyter notebooks for OR/Optimization problems
- `data/` - Datasets used in notebooks
- `scripts/` - Standalone Python scripts for data processing/analysis