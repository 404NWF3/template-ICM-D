"""
Simulated Annealing Framework for MCM/ICM Optimization Problems
Production-ready implementation with various cooling schedules.
"""

import numpy as np
from typing import Callable, Optional, Tuple
import math


class SimulatedAnnealing:
    """
    Generic Simulated Annealing for combinatorial optimization.
    Supports various cooling schedules and neighbor generation strategies.
    """

    def __init__(self,
                 objective_func: Callable,
                 initial_solution: np.ndarray,
                 objective_type: str = 'minimize',
                 temperature_schedule: str = 'exponential',
                 initial_temp: float = 1000.0,
                 final_temp: float = 0.01,
                 cooling_rate: float = 0.95,
                 iterations_per_temp: int = 100,
                 max_iterations: int = 10000):
        """
        Initialize SA parameters.

        Args:
            objective_func: Function to evaluate solution quality
            initial_solution: Starting solution
            objective_type: 'minimize' or 'maximize'
            temperature_schedule: 'exponential', 'logarithmic', or 'linear'
            initial_temp: Starting temperature
            final_temp: Stopping temperature
            cooling_rate: Temperature reduction factor
            iterations_per_temp: Evaluations at each temperature
            max_iterations: Maximum total iterations
        """
        self.objective_func = objective_func
        self.current_solution = initial_solution.copy()
        self.objective_type = objective_type
        self.temperature_schedule = temperature_schedule
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.max_iterations = max_iterations

        # History
        self.solution_history = []
        self.objective_history = []
        self.temperature_history = []
        self.acceptance_history = []

        # Initialize
        self.current_objective = self.objective_func(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_objective = self.current_objective

    def temperature(self, iteration: int) -> float:
        """Calculate temperature at given iteration."""
        if self.temperature_schedule == 'exponential':
            return self.initial_temp * (self.cooling_rate ** iteration)
        elif self.temperature_schedule == 'logarithmic':
            return self.initial_temp / (1 + self.cooling_rate * np.log(1 + iteration))
        elif self.temperature_schedule == 'linear':
            return max(self.final_temp, self.initial_temp - (self.initial_temp - self.final_temp) *
                       (iteration / self.max_iterations))
        else:
            raise ValueError(f"Unknown schedule: {self.temperature_schedule}")

    def acceptance_probability(self, delta: float, temp: float) -> float:
        """Calculate probability of accepting worse solution."""
        if self.objective_type == 'minimize':
            # delta = new - old (positive means worse)
            return math.exp(-delta / temp) if temp > 0 else 0
        else:
            # Maximization: negative delta means worse
            return math.exp(delta / temp) if temp > 0 else 0

    def generate_neighbor(self, solution: np.ndarray, perturbation: str = 'swap') -> np.ndarray:
        """Generate neighboring solution."""
        neighbor = solution.copy()

        if perturbation == 'swap' and len(neighbor.shape) == 1:
            # Swap two random elements (for permutations)
            i, j = np.random.choice(len(neighbor), 2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        elif perturbation == 'bit_flip' and len(neighbor.shape) == 1:
            # Flip random bits (for binary)
            num_flips = max(1, len(neighbor) // 10)
            indices = np.random.choice(len(neighbor), num_flips, replace=False)
            neighbor[indices] = 1 - neighbor[indices]

        elif perturbation == 'adjacent_swap' and len(neighbor.shape) == 1:
            # Swap adjacent elements (for permutations, smoother search)
            i = np.random.choice(len(neighbor) - 1)
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]

        elif perturbation == 'insertion' and len(neighbor.shape) == 1:
            # Remove and insert at different position (for permutations)
            i = np.random.choice(len(neighbor))
            j = np.random.choice(len(neighbor))
            element = neighbor[i]
            neighbor = np.delete(neighbor, i)
            neighbor = np.insert(neighbor, j, element)

        else:
            # Default: small random perturbation
            noise = np.random.normal(0, 0.1, neighbor.shape)
            neighbor = neighbor + noise

        return neighbor

    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run simulated annealing.

        Returns:
            (best_solution, best_objective)
        """
        temp = self.initial_temp
        iteration = 0
        total_accepted = 0

        while temp > self.final_temp and iteration < self.max_iterations:
            accepted_this_temp = 0

            for _ in range(self.iterations_per_temp):
                # Generate neighbor
                neighbor = self.generate_neighbor(self.current_solution)
                neighbor_objective = self.objective_func(neighbor)

                # Calculate change
                delta = neighbor_objective - self.current_objective

                # Accept or reject
                if delta < 0 or np.random.rand() < self.acceptance_probability(delta, temp):
                    self.current_solution = neighbor
                    self.current_objective = neighbor_objective
                    accepted_this_temp += 1

                    # Update best
                    if self.objective_type == 'minimize':
                        if neighbor_objective < self.best_objective:
                            self.best_solution = neighbor.copy()
                            self.best_objective = neighbor_objective
                    else:
                        if neighbor_objective > self.best_objective:
                            self.best_solution = neighbor.copy()
                            self.best_objective = neighbor_objective

                iteration += 1
                if iteration >= self.max_iterations:
                    break

            # Record history
            self.solution_history.append(self.current_solution.copy())
            self.objective_history.append(self.current_objective)
            self.temperature_history.append(temp)
            total_accepted += accepted_this_temp
            acceptance_rate = accepted_this_temp / self.iterations_per_temp
            self.acceptance_history.append(acceptance_rate)

            # Update temperature
            temp = self.temperature(iteration)

            if verbose and iteration % (self.iterations_per_temp * 10) == 0:
                print(f"Iter {iteration}: Temp={temp:.4f}, "
                      f"Current={self.current_objective:.4f}, "
                      f"Best={self.best_objective:.4f}, "
                      f"Accept={acceptance_rate:.2%}")

        if verbose:
            total_acceptance_rate = total_accepted / iteration
            print(f"\nCompleted {iteration} iterations")
            print(f"Final temperature: {temp:.6f}")
            print(f"Overall acceptance rate: {total_acceptance_rate:.2%}")
            print(f"Best objective: {self.best_objective:.4f}")

        return self.best_solution, self.best_objective


# Example: TSP with Simulated Annealing

def tsp_objective(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Calculate total tour distance."""
    total = 0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]
        total += distance_matrix[from_city, to_city]
    return total


def create_tsp_objective(distance_matrix: np.ndarray) -> Callable:
    """Factory for TSP objective function."""
    return lambda tour: tsp_objective(tour, distance_matrix)


def solve_tsp_sa(distance_matrix: np.ndarray,
                 initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995,
                 verbose: bool = True) -> Tuple[np.ndarray, float]:
    """
    Solve TSP using simulated annealing.

    Args:
        distance_matrix: Square matrix of distances
        initial_temp: Starting temperature
        cooling_rate: Exponential cooling factor
        verbose: Print progress

    Returns:
        (best_tour, best_distance)
    """
    n_cities = len(distance_matrix)
    initial_tour = np.random.permutation(n_cities)

    objective = create_tsp_objective(distance_matrix)

    sa = SimulatedAnnealing(
        objective_func=objective,
        initial_solution=initial_tour,
        objective_type='minimize',
        temperature_schedule='exponential',
        initial_temp=initial_temp,
        final_temp=0.01,
        cooling_rate=cooling_rate,
        iterations_per_temp=max(10, n_cities),
        max_iterations=100000
    )

    return sa.solve(verbose=verbose)


# Example: Knapsack with Simulated Annealing

def knapsack_objective(solution: np.ndarray, values: np.ndarray,
                       weights: np.ndarray, capacity: float,
                       penalty: float = 1000) -> float:
    """
    Calculate knapsack value (with penalty for over-capacity).
    Negative because SA minimizes by default.
    """
    total_value = np.dot(solution, values)
    total_weight = np.dot(solution, weights)

    if total_weight > capacity:
        penalty_cost = penalty * (total_weight - capacity)
        return -(total_value - penalty_cost)
    return -total_value


def create_knapsack_objective(values: np.ndarray, weights: np.ndarray,
                              capacity: float, penalty: float = 1000) -> Callable:
    """Factory for knapsack objective function."""
    return lambda sol: knapsack_objective(sol, values, weights, capacity, penalty)


def solve_knapsack_sa(values: np.ndarray, weights: np.ndarray, capacity: float,
                     initial_temp: float = 100.0, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """
    Solve knapsack using simulated annealing.

    Returns:
        (best_solution, best_value)
    """
    initial_solution = np.random.randint(0, 2, len(values))
    objective = create_knapsack_objective(values, weights, capacity)

    sa = SimulatedAnnealing(
        objective_func=objective,
        initial_solution=initial_solution,
        objective_type='minimize',
        temperature_schedule='exponential',
        initial_temp=initial_temp,
        final_temp=0.01,
        cooling_rate=0.99,
        iterations_per_temp=50,
        max_iterations=10000
    )

    solution, obj = sa.solve(verbose=verbose)
    return solution, -obj  # Convert back to positive value
