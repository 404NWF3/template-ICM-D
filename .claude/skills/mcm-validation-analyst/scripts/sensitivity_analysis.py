"""
Sensitivity Analysis Tools for MCM/ICM Optimization Models
Generate visualizations and metrics for parameter sensitivity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Optional
import itertools


def one_way_sensitivity(base_params: Dict, param_name: str,
                       param_values: np.ndarray,
                       objective_func: Callable,
                       solver_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one-way sensitivity analysis on a single parameter.

    Args:
        base_params: Dictionary of baseline parameter values
        param_name: Name of parameter to vary
        param_values: Array of parameter values to test
        objective_func: Function to compute objective value
        solver_func: Function to solve the optimization problem

    Returns:
        (param_values_tested, objective_values)
    """
    objective_values = []

    for value in param_values:
        # Update parameter
        test_params = base_params.copy()
        test_params[param_name] = value

        # Solve and get objective
        solution = solver_func(test_params)
        obj_value = objective_func(solution, test_params)
        objective_values.append(obj_value)

    return np.array(param_values), np.array(objective_values)


def two_way_sensitivity(base_params: Dict, param1_name: str, param1_values: np.ndarray,
                        param2_name: str, param2_values: np.ndarray,
                        objective_func: Callable, solver_func: Callable) -> np.ndarray:
    """
    Perform two-way sensitivity analysis.

    Returns:
        2D array of objective values
    """
    n1, n2 = len(param1_values), len(param2_values)
    objective_matrix = np.zeros((n2, n1))

    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            test_params = base_params.copy()
            test_params[param1_name] = val1
            test_params[param2_name] = val2

            solution = solver_func(test_params)
            objective_matrix[j, i] = objective_func(solution, test_params)

    return objective_matrix


def compute_elasticity(param_values: np.ndarray, objective_values: np.ndarray) -> np.ndarray:
    """
    Compute elasticity: % change in objective / % change in parameter.

    Returns array of elasticities (excluding base point).
    """
    base_idx = len(param_values) // 2
    base_param = param_values[base_idx]
    base_obj = objective_values[base_idx]

    elasticities = []

    for i, (p_val, o_val) in enumerate(zip(param_values, objective_values)):
        if i != base_idx and base_param != 0 and base_obj != 0:
            pct_change_param = (p_val - base_param) / abs(base_param)
            pct_change_obj = (o_val - base_obj) / abs(base_obj)
            elasticities.append(pct_change_obj / pct_change_param)

    return np.array(elasticities)


def plot_one_way_sensitivity(param_values: np.ndarray, objective_values: np.ndarray,
                             param_name: str, param_label: str,
                             save_path: Optional[str] = None) -> None:
    """Plot one-way sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main line
    ax.plot(param_values, objective_values, 'b-', linewidth=2, label='Objective Value')

    # Base case marker
    base_idx = len(param_values) // 2
    ax.axvline(param_values[base_idx], color='r', linestyle='--',
               alpha=0.7, label='Base Case')
    ax.scatter([param_values[base_idx]], [objective_values[base_idx]],
               color='r', s=100, zorder=5)

    # Range indicators
    min_obj, max_obj = objective_values.min(), objective_values.max()
    ax.fill_between(param_values, min_obj, max_obj, alpha=0.2, color='blue')

    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title(f'One-Way Sensitivity: {param_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'sensitivity_{param_name.replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
    plt.close()


def plot_two_way_heatmap(param1_values: np.ndarray, param2_values: np.ndarray,
                        objective_matrix: np.ndarray, param1_label: str,
                        param2_label: str, title: str = 'Two-Way Sensitivity Analysis',
                        save_path: Optional[str] = None) -> None:
    """Plot two-way sensitivity heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    im = ax.imshow(objective_matrix, origin='lower', aspect='auto',
                   extent=[param1_values.min(), param1_values.max(),
                          param2_values.min(), param2_values.max()],
                   cmap='RdYlGn_r')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Objective Value', fontsize=11)

    # Base case marker
    base1_idx = len(param1_values) // 2
    base2_idx = len(param2_values) // 2
    ax.scatter([param1_values[base1_idx]], [param2_values[base2_idx]],
               marker='x', s=200, color='blue', linewidths=3,
               label='Base Case', zorder=5)

    ax.set_xlabel(param1_label, fontsize=12)
    ax.set_ylabel(param2_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_tornado_chart(params_dict: Dict[str, np.ndarray],
                      base_objective: float,
                      save_path: Optional[str] = None) -> None:
    """
    Plot tornado chart showing parameter impact magnitude.

    Args:
        params_dict: {param_name: array_of_objective_values}
        base_objective: Baseline objective value
    """
    # Calculate impact ranges
    param_names = list(params_dict.keys())
    impacts = []
    low_values = []
    high_values = []

    for name, values in params_dict.items():
        min_val = values.min()
        max_val = values.max()
        impact = max_val - min_val
        impacts.append(impact)
        low_values.append(min_val)
        high_values.append(max_val)

    # Sort by impact magnitude
    sorted_idx = np.argsort(impacts)[::-1]
    param_names_sorted = [param_names[i] for i in sorted_idx]
    impacts_sorted = [impacts[i] for i in sorted_idx]
    low_sorted = [low_values[i] for i in sorted_idx]
    high_sorted = [high_values[i] for i in sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(param_names) * 0.4)))

    y_pos = np.arange(len(param_names_sorted))

    # Horizontal bars showing ranges
    for i, (low, high) in enumerate(zip(low_sorted, high_sorted)):
        ax.barh(y_pos[i], high - low, left=low, height=0.6,
               color='steelblue', alpha=0.7)

    # Base case line
    ax.axvline(base_objective, color='r', linestyle='--', linewidth=2, label='Base Case')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names_sorted)
    ax.set_xlabel('Objective Value', fontsize=12)
    ax.set_title('Tornado Diagram: Parameter Impact Analysis',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('tornado_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_spider_chart(param_names: List[str], normalized_impacts: List[float],
                     save_path: Optional[str] = None) -> None:
    """Plot spider/radar chart for multi-parameter sensitivity."""
    n_params = len(param_names)
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    normalized_impacts = normalized_impacts + normalized_impacts[:1]  # Close the loop
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    ax.plot(angles, normalized_impacts, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, normalized_impacts, alpha=0.25, color='steelblue')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(param_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)

    plt.title('Parameter Sensitivity Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('spider_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_sensitivity_report(base_params: Dict, analysis_results: Dict,
                               filename: str = 'sensitivity_report.txt') -> None:
    """Generate text report of sensitivity analysis."""
    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SENSITIVITY ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("BASELINE PARAMETERS:\n")
        for key, value in base_params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "-" * 60 + "\n\n")

        for param_name, results in analysis_results.items():
            f.write(f"PARAMETER: {param_name}\n")
            f.write(f"  Tested values: {results['values']}\n")
            f.write(f"  Objective range: [{results['obj_min']:.4f}, {results['obj_max']:.4f}]\n")
            f.write(f"  Total variation: {results['obj_max'] - results['obj_min']:.4f}\n")
            f.write(f"  Normalized range: {(results['obj_max'] - results['obj_min']) / abs(results['base']):.2%}\n")

            # Elasticity
            if 'elasticities' in results:
                mean_elasticity = np.mean(results['elasticities'])
                f.write(f"  Mean elasticity: {mean_elasticity:.4f}\n")

            # Classification
            variation_pct = (results['obj_max'] - results['obj_min']) / abs(results['base'])
            if variation_pct < 0.05:
                classification = "ROBUST"
            elif variation_pct < 0.20:
                classification = "MODERATELY SENSITIVE"
            else:
                classification = "HIGHLY SENSITIVE"

            f.write(f"  Classification: {classification}\n")
            f.write("\n")


# Example usage

if __name__ == "__main__":
    # Example: Simple linear program sensitivity

    def example_objective(solution, params):
        return solution['objective']

    def example_solver(params):
        # Dummy solver - replace with actual optimization
        cost = params.get('cost', 10)
        demand = params.get('demand', 100)
        capacity = params.get('capacity', 150)

        # Simple objective calculation
        obj = cost * min(demand, capacity)
        return {'objective': obj, 'x': min(demand, capacity)}

    # Base parameters
    base_params = {'cost': 10.0, 'demand': 100.0, 'capacity': 150.0}

    # One-way sensitivity on cost
    cost_values = np.linspace(5, 15, 20)
    cost_tested, cost_objectives = one_way_sensitivity(
        base_params, 'cost', cost_values, example_objective, example_solver
    )

    plot_one_way_sensitivity(cost_tested, cost_objectives, 'Unit Cost',
                            'Cost per Unit ($)')

    # Tornado chart for all parameters
    all_results = {}

    for param in ['cost', 'demand', 'capacity']:
        if param == 'cost':
            values = np.linspace(5, 15, 15)
        elif param == 'demand':
            values = np.linspace(50, 150, 15)
        else:
            values = np.linspace(100, 200, 15)

        _, objs = one_way_sensitivity(base_params, param, values,
                                     example_objective, example_solver)
        all_results[param] = objs

    base_obj = example_solver(base_params)['objective']
    plot_tornado_chart(all_results, base_obj)

    print("Sensitivity analysis complete. Plots saved.")
