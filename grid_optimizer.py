"""
Improved Distribution Grid Optimization using Genetic Algorithm
"""
import pandas as pd
import numpy as np
import random
from typing import Tuple, List
from utils import run_time_series


class GridOptimizer:
    """
    Genetic Algorithm optimizer for DER and load allocation in distribution grids.
    """

    def __init__(
        self,
        gen_data: pd.DataFrame,
        load_data: pd.DataFrame,
        net,
        population_size: int = 20,
        mutation_rate: float = 0.15,
        elite_size: int = 2,
        max_generations: int = 50,
        convergence_threshold: float = 0.001,
        verbose: bool = True
    ):
        """
        Initialize the grid optimizer.

        Parameters:
        -----------
        gen_data : pd.DataFrame
            Generation data for all households
        load_data : pd.DataFrame
            Load data for all households
        net : pandapower network
            The pandapower network object
        population_size : int
            Number of individuals in population
        mutation_rate : float
            Probability of mutation (0-1)
        elite_size : int
            Number of best individuals to preserve
        max_generations : int
            Maximum number of generations to run
        convergence_threshold : float
            Stop if improvement is less than this threshold
        verbose : bool
            Print progress information
        """
        self.gen_data = gen_data
        self.load_data = load_data
        self.net = net
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose

        self.n_households = len(gen_data.columns)

        # Best solution tracking
        self.best_gen_order = None
        self.best_load_order = None
        self.best_fitness = float('inf')
        self.fitness_history = []

    def evaluate_fitness(self, gen_order: List[int], load_order: List[int]) -> float:
        """
        Evaluate fitness of a solution (lower is better).

        Returns sum of maximum line loading across all time steps.
        """
        try:
            gen_order_idx = pd.Index(gen_order)
            load_order_idx = pd.Index(load_order)

            res_ext, res_lines = run_time_series(
                self.gen_data,
                self.load_data,
                self.net,
                index_order_gen=gen_order_idx,
                index_order_load=load_order_idx
            )

            # Fitness = sum of max loading per time step (lower is better)
            fitness = res_lines.max(axis=1).sum()
            return fitness

        except Exception as e:
            if self.verbose:
                print(f"Error evaluating fitness: {e}")
            return float('inf')

    def initialize_population(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Create initial random population.
        """
        gen_population = []
        load_population = []

        base_order = list(range(self.n_households))

        for _ in range(self.population_size):
            gen_order = base_order.copy()
            load_order = base_order.copy()

            random.shuffle(gen_order)
            random.shuffle(load_order)

            gen_population.append(gen_order)
            load_population.append(load_order)

        return gen_population, load_population

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Order crossover (OX) - preserves relative order and ensures all elements present.
        """
        size = len(parent1)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)

        # Take substring from parent1
        child = [-1] * size
        child[start:end] = parent1[start:end]

        # Fill remaining positions from parent2
        current_pos = end % size
        for gene in parent2:
            if gene not in child:
                child[current_pos] = gene
                current_pos = (current_pos + 1) % size

        return child

    def mutate(self, order: List[int]) -> List[int]:
        """
        Swap mutation - randomly swap two positions.
        """
        order = order.copy()

        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(order)), 2)
            order[idx1], order[idx2] = order[idx2], order[idx1]

        return order

    def select_parents(
        self,
        population_gen: List[List[int]],
        population_load: List[List[int]],
        fitness_scores: List[float]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Tournament selection - select 2 parents.
        """
        tournament_size = 3

        def tournament():
            tournament_indices = random.sample(range(len(population_gen)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            return winner_idx

        parent1_idx = tournament()
        parent2_idx = tournament()

        return (
            population_gen[parent1_idx],
            population_load[parent1_idx],
            population_gen[parent2_idx],
            population_load[parent2_idx]
        )

    def evolve_generation(
        self,
        population_gen: List[List[int]],
        population_load: List[List[int]],
        fitness_scores: List[float]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Create next generation using elitism, crossover, and mutation.
        """
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(fitness_scores)

        new_pop_gen = []
        new_pop_load = []

        # Elitism - keep best individuals
        for i in range(self.elite_size):
            idx = sorted_indices[i]
            new_pop_gen.append(population_gen[idx].copy())
            new_pop_load.append(population_load[idx].copy())

        # Create offspring
        while len(new_pop_gen) < self.population_size:
            # Select parents
            p1_gen, p1_load, p2_gen, p2_load = self.select_parents(
                population_gen, population_load, fitness_scores
            )

            # Crossover
            child_gen = self.crossover(p1_gen, p2_gen)
            child_load = self.crossover(p1_load, p2_load)

            # Mutation
            child_gen = self.mutate(child_gen)
            child_load = self.mutate(child_load)

            new_pop_gen.append(child_gen)
            new_pop_load.append(child_load)

        return new_pop_gen[:self.population_size], new_pop_load[:self.population_size]

    def optimize(self) -> Tuple[List[int], List[int], float]:
        """
        Run genetic algorithm optimization.

        Returns:
        --------
        best_gen_order : List[int]
            Optimal generator allocation
        best_load_order : List[int]
            Optimal load allocation
        best_fitness : float
            Best fitness value achieved
        """
        if self.verbose:
            print(f"Starting genetic algorithm optimization...")
            print(f"Population size: {self.population_size}")
            print(f"Max generations: {self.max_generations}")
            print(f"Mutation rate: {self.mutation_rate}")
            print("-" * 60)

        # Initialize population
        population_gen, population_load = self.initialize_population()

        # Track best solution
        self.best_fitness = float('inf')
        generations_without_improvement = 0

        for generation in range(self.max_generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for gen_order, load_order in zip(population_gen, population_load):
                fitness = self.evaluate_fitness(gen_order, load_order)
                fitness_scores.append(fitness)

            # Find best in current generation
            min_fitness_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[min_fitness_idx]

            # Update global best
            improvement = self.best_fitness - current_best_fitness
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_gen_order = population_gen[min_fitness_idx].copy()
                self.best_load_order = population_load[min_fitness_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            self.fitness_history.append(current_best_fitness)

            if self.verbose and generation % 5 == 0:
                print(f"Generation {generation:3d}: Best Fitness = {current_best_fitness:.4f}, "
                      f"Improvement = {improvement:.6f}")

            # Check convergence
            if improvement < self.convergence_threshold and generations_without_improvement >= 10:
                if self.verbose:
                    print(f"\nConverged at generation {generation}")
                break

            # Evolve to next generation
            population_gen, population_load = self.evolve_generation(
                population_gen, population_load, fitness_scores
            )

        if self.verbose:
            print("-" * 60)
            print(f"Optimization complete!")
            print(f"Best fitness: {self.best_fitness:.4f}")
            print(f"Best gen order: {self.best_gen_order}")
            print(f"Best load order: {self.best_load_order}")

        return self.best_gen_order, self.best_load_order, self.best_fitness

    def plot_fitness_history(self):
        """
        Plot fitness evolution over generations.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness (Total Line Loading)', fontsize=12)
        plt.title('Genetic Algorithm Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('optimization_history.png', dpi=150)
        plt.show()

        return plt.gcf()
