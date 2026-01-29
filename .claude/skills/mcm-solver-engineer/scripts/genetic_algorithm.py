"""
Genetic Algorithm Framework for MCM/ICM Optimization Problems
Production-ready implementation with multiple operators and convergence detection.
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
import copy


class GeneticAlgorithm:
    """
    Generic Genetic Algorithm for permutation and binary optimization.
    Supports various crossover and mutation operators.
    """

    def __init__(self,
                 fitness_func: Callable,
                 chromosome_type: str = 'permutation',
                 pop_size: int = 100,
                 elite_size: int = 5,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 3,
                 max_generations: int = 500,
                 convergence_threshold: int = 50):
        """
        Initialize GA parameters.

        Args:
            fitness_func: Function to evaluate chromosome fitness (lower is better)
            chromosome_type: 'permutation', 'binary', or 'custom'
            pop_size: Population size
            elite_size: Number of elite individuals preserved
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
            max_generations: Maximum iterations
            convergence_threshold: Generations without improvement before stopping
        """
        self.fitness_func = fitness_func
        self.chromosome_type = chromosome_type
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold

        # History tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def initialize_population(self, n_genes: int) -> np.ndarray:
        """Create initial population."""
        if self.chromosome_type == 'permutation':
            pop = np.array([np.random.permutation(n_genes) for _ in range(self.pop_size)])
        elif self.chromosome_type == 'binary':
            pop = np.random.randint(0, 2, (self.pop_size, n_genes))
        else:
            raise ValueError(f"Unknown chromosome type: {self.chromosome_type}")
        return pop

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for all individuals."""
        fitness = np.array([self.fitness_func(ind) for ind in population])
        return fitness

    def tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select parents using tournament selection."""
        selected = []
        for _ in range(self.pop_size - self.elite_size):
            # Randomly select tournament_size individuals
            tournament_idx = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitness = fitness[tournament_idx]
            # Select best from tournament
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return np.array(selected)

    def order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Order crossover (OX1) for permutation problems."""
        size = len(parent1)
        child1, child2 = np.full(size, -1), np.full(size, -1)

        # Select random subset
        start, end = sorted(np.random.choice(range(size), 2, replace=False))

        # Copy subset to children
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # Fill remaining from other parent
        def fill_child(child, other_parent):
            remaining = [x for x in other_parent if x not in child]
            idx = 0
            for i in range(size):
                if child[i] == -1:
                    child[i] = remaining[idx]
                    idx += 1

        fill_child(child1, parent2)
        fill_child(child2, parent1)

        return child1, child2

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover for binary problems."""
        mask = np.random.rand(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply crossover based on chromosome type."""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        if self.chromosome_type == 'permutation':
            return self.order_crossover(parent1, parent2)
        elif self.chromosome_type == 'binary':
            return self.uniform_crossover(parent1, parent2)
        else:
            return parent1.copy(), parent2.copy()

    def swap_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Swap two genes (for permutations)."""
        mutant = chromosome.copy()
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(len(mutant), 2, replace=False)
            mutant[i], mutant[j] = mutant[j], mutant[i]
        return mutant

    def bit_flip_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Flip bits (for binary)."""
        mutant = chromosome.copy()
        mask = np.random.rand(len(mutant)) < self.mutation_rate
        mutant[mask] = 1 - mutant[mask]
        return mutant

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Apply mutation based on chromosome type."""
        if self.chromosome_type == 'permutation':
            return self.swap_mutation(chromosome)
        elif self.chromosome_type == 'binary':
            return self.bit_flip_mutation(chromosome)
        else:
            return chromosome.copy()

    def evolve(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Create next generation."""
        # Sort by fitness (ascending - lower is better)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        # Elitism: preserve best individuals
        new_pop = population[:self.elite_size].copy()

        # Select parents
        parents = self.tournament_selection(population, fitness)

        # Create offspring
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))
            else:
                offspring.append(self.mutate(parents[i]))

        new_pop = np.vstack([new_pop] + offspring)
        return new_pop

    def check_convergence(self, generation: int, best_fitness: float) -> bool:
        """Check if algorithm should converge."""
        if len(self.best_fitness_history) < self.convergence_threshold:
            return False

        recent_best = self.best_fitness_history[-self.convergence_threshold:]
        return all(abs(f - best_fitness) < 1e-6 for f in recent_best)

    def solve(self, n_genes: int, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run the genetic algorithm.

        Returns:
            (best_solution, best_fitness)
        """
        # Initialize
        population = self.initialize_population(n_genes)
        fitness = self.evaluate_population(population)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for gen in range(self.max_generations):
            # Track history
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness))

            # Check convergence
            if self.check_convergence(gen, best_fitness):
                if verbose:
                    print(f"Converged at generation {gen}")
                break

            # Evolve
            population = self.evolve(population, fitness)
            fitness = self.evaluate_population(population)

            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()

            if verbose and gen % 50 == 0:
                print(f"Gen {gen}: Best={best_fitness:.4f}, Avg={np.mean(fitness):.4f}")

        return best_solution, best_fitness


# Example usage functions

def example_tsp_fitness(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Calculate total distance for TSP tour."""
    total = 0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]
        total += distance_matrix[from_city, to_city]
    return total


def create_tsp_fitness(distance_matrix: np.ndarray) -> Callable:
    """Factory function to create TSP fitness function."""
    def fitness(tour: np.ndarray) -> float:
        return example_tsp_fitness(tour, distance_matrix)
    return fitness


def example_knapsack_fitness(solution: np.ndarray, values: np.ndarray,
                              weights: np.ndarray, capacity: float,
                              penalty: float = 1000) -> float:
    """Calculate fitness for knapsack (maximize value, minimize penalty for over-capacity)."""
    total_value = np.dot(solution, values)
    total_weight = np.dot(solution, weights)
    penalty_cost = penalty * max(0, total_weight - capacity)
    return -(total_value - penalty_cost)  # Negative for minimization
