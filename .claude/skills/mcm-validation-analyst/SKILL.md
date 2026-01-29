---
name: mcm-validation-analyst
description: Perform sensitivity analysis and validation for MCM/ICM optimization models. Use when Claude needs to: (1) Design sensitivity analysis plans (parameter selection, variation ranges), (2) Generate Python code for sensitivity visualization (line plots, heatmaps, tornado charts), (3) Provide robustness recommendations for parameter-sensitive solutions, (4) Interpret optimal solutions in practical contexts (logistics, production, etc.), or (5) Explain optimality gaps and comparative advantages
---

# MCM Validation Analyst

Validate model reliability through sensitivity analysis and practical interpretation.

## Sensitivity Analysis Design

### Parameter Selection Strategy

Identify critical parameters to test:

1. **Objective coefficients**: Costs, prices, weights
2. **Right-hand side values**: Resource limits, demands, capacities
3. **Technical coefficients**: Consumption rates, efficiency factors
4. **External parameters**: Market conditions, environmental factors

**Selection criteria:**
- High uncertainty in true value
- Large impact on solution structure
- Policy-relevant parameters

### Variation Range Planning

```python
# Define parameter ranges to test
parameter_ranges = {
    'cost_per_unit': {'base': 10.0, 'range': (5.0, 15.0), 'steps': 20},
    'capacity_limit': {'base': 1000, 'range': (800, 1200), 'steps': 15},
    'demand_growth': {'base': 0.05, 'range': (0.0, 0.10), 'steps': 25}
}
```

**Guidelines:**
- Use ±20-50% for well-known parameters
- Use ±100% or more for uncertain parameters
- Include 0 to test boundary behavior
- Use at least 10-20 evaluation points for smooth plots

## Visualization Generation

See [scripts/sensitivity_analysis.py](scripts/sensitivity_analysis.py) for complete implementation.

### One-Way Sensitivity Plot

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_one_way_sensitivity(param_name, param_values, objective_values):
    """Plot objective vs single parameter variation."""
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, objective_values, linewidth=2)
    plt.axvline(param_values[len(param_values)//2], linestyle='--',
                color='red', label='Base case')
    plt.xlabel(f'{param_name}', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.title(f'Sensitivity Analysis: {param_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'sensitivity_{param_name}.png', dpi=300)
    plt.close()
```

### Two-Way Sensitivity Heatmap

```python
def plot_two_way_heatmap(param1_values, param2_values, objective_matrix,
                         param1_name, param2_name):
    """Heatmap showing objective over two parameters."""
    plt.figure(figsize=(10, 8))
    plt.imshow(objective_matrix, origin='lower', aspect='auto',
               extent=[param1_values.min(), param1_values.max(),
                      param2_values.min(), param2_values.max()],
               cmap='RdYlGn_r')
    plt.colorbar(label='Objective Value')
    plt.xlabel(param1_name, fontsize=12)
    plt.ylabel(param2_name, fontsize=12)
    plt.title('Two-Way Sensitivity Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('sensitivity_heatmap.png', dpi=300)
    plt.close()
```

### Tornado Diagram

```python
def plot_tornado_chart(params_dict, base_objective):
    """Tornado chart showing parameter impact magnitude."""
    params = list(params_dict.keys())
    impacts = [abs(max(v) - min(v)) for v in params_dict.values()]

    sorted_idx = np.argsort(impacts)[::-1]
    params_sorted = [params[i] for i in sorted_idx]
    impacts_sorted = [impacts[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(params_sorted)), impacts_sorted)
    plt.yticks(range(len(params_sorted)), params_sorted)
    plt.xlabel('Objective Value Range', fontsize=12)
    plt.title('Tornado Diagram: Parameter Impact', fontsize=14)
    plt.tight_layout()
    plt.savefig('tornado_diagram.png', dpi=300)
    plt.close()
```

## Robustness Assessment

### Sensitivity Metrics

```python
def compute_sensitivity_metrics(param_values, objective_values):
    """Calculate sensitivity indicators."""
    base_idx = len(param_values) // 2
    base_value = objective_values[base_idx]

    metrics = {
        'elasticity': [],  # % change in objective / % change in param
        'max_change': max(abs(v - base_value) for v in objective_values),
        'normalized_range': (max(objective_values) - min(objective_values)) /
                           abs(base_value) if base_value != 0 else 0
    }

    for i, val in enumerate(param_values):
        if i != base_idx and param_values[base_idx] != 0:
            pct_change_param = (val - param_values[base_idx]) / param_values[base_idx]
            pct_change_obj = (objective_values[i] - base_value) / abs(base_value)
            metrics['elasticity'].append(pct_change_obj / pct_change_param)

    return metrics
```

### Robustness Classification

- **Robust**: < 5% objective change for ±50% parameter variation
- **Moderately sensitive**: 5-20% objective change
- **Highly sensitive**: > 20% objective change

### Enhancement Strategies

For sensitive solutions:

1. **Conservative planning**: Use worst-case parameter values
2. **Hedging**: Maintain slack resources
3. **Diversification**: Spread investment across options
4. **Stochastic programming**: Explicitly model parameter distributions
5. **Robust optimization**: Optimize for worst-case scenario

## Solution Interpretation

### Physical Meaning Framework

```python
def interpret_solution(solution, problem_context):
    """Generate practical interpretation."""
    interpretation = {
        'what': 'What the numbers represent',
        'why': 'Why this solution is optimal',
        'how': 'How to implement in practice',
        'tradeoffs': 'Key tradeoffs made'
    }
    return interpretation
```

### Comparative Analysis

**vs Baseline:**
- Naive approach: greedy, random, or rule-based
- Calculate improvement percentage
- Explain source of advantage (better coordination, risk pooling, etc.)

**vs Alternative:**
- Compare with second-best solution
- Analyze why optimal is preferred
- Discuss when alternative might be better

## Validation Checklist

- [ ] Solution satisfies all constraints (numerical check)
- [ ] Solution makes practical sense (sanity check)
- [ ] Key parameters tested for sensitivity
- [ ] Visualization shows clear patterns
- [ ] Interpretation links to problem context
- [ ] Limitations acknowledged
- [ ] Robustness recommendations provided

## Common Pitfalls

1. **Testing only optimistic scenarios**: Include both favorable and adverse cases
2. **Ignoring correlation**: Parameters often vary together
3. **Overfitting to data**: Model may not generalize to new conditions
4. **Hidden constraints**: Practical implementation may have unstated limits
5. **Static analysis**: Real-world systems evolve over time
