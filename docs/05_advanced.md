# Advanced Genetic Algorithm Techniques

This document covers advanced concepts, techniques, and optimizations for genetic algorithms. These topics are for users who want to push the boundaries of what's possible with evolutionary computation.

## Table of Contents

1. [Hybrid Algorithms](#hybrid-algorithms)
2. [Multi-Objective Optimization](#multi-objective-optimization)
3. [Parallel and Distributed GAs](#parallel-and-distributed-gas)
4. [Adaptive Parameter Control](#adaptive-parameter-control)
5. [Advanced Selection Strategies](#advanced-selection-strategies)
6. [Constraint Handling Techniques](#constraint-handling-techniques)
7. [Population Diversity Maintenance](#population-diversity-maintenance)
8. [Algorithm Performance Analysis](#algorithm-performance-analysis)
9. [Real-World Deployment Considerations](#real-world-deployment-considerations)

## Hybrid Algorithms

Combining genetic algorithms with other optimization techniques often yields superior results.

### GA + Local Search (Memetic Algorithms)

```python
from genetic_algorithms.core.individual import Individual
import numpy as np
import random

class MemeticIndividual(Individual):
    """Individual that applies local search after genetic operations."""
    
    def __init__(self, genes=None, **kwargs):
        self.local_search_prob = kwargs.get('local_search_prob', 0.1)
        self.local_search_intensity = kwargs.get('local_search_intensity', 10)
        super().__init__(genes, **kwargs)
    
    def apply_local_search(self):
        """Apply local search to improve the individual."""
        if random.random() > self.local_search_prob:
            return self
        
        best_neighbor = self
        best_fitness = self.fitness
        
        # Try random neighbors
        for _ in range(self.local_search_intensity):
            neighbor = self.get_random_neighbor()
            if neighbor.fitness > best_fitness:
                best_neighbor = neighbor
                best_fitness = neighbor.fitness
        
        return best_neighbor
    
    def get_random_neighbor(self):
        """Generate a random neighbor for local search."""
        # Implementation depends on problem representation
        # Example for real-valued problems:
        new_genes = self.genes.copy()
        if isinstance(new_genes[0], (int, float)):
            # Add small random perturbation
            idx = random.randint(0, len(new_genes) - 1)
            perturbation = np.random.normal(0, 0.1)
            new_genes[idx] += perturbation
        
        return self.__class__(new_genes, **self.kwargs)

# Modified GA with local search
class MemeticGA(GeneticAlgorithm):
    """Genetic Algorithm with local search (Memetic Algorithm)."""
    
    def evolve_generation(self):
        """Override to include local search step."""
        # Standard GA operations
        super().evolve_generation()
        
        # Apply local search to some individuals
        improved_individuals = []
        for individual in self.population.individuals:
            if hasattr(individual, 'apply_local_search'):
                improved = individual.apply_local_search()
                improved_individuals.append(improved)
            else:
                improved_individuals.append(individual)
        
        self.population.individuals = improved_individuals
        self.population.evaluate_fitness()
```

### GA + Simulated Annealing

```python
class SAHybridIndividual(Individual):
    """Individual that uses Simulated Annealing for local improvement."""
    
    def __init__(self, genes=None, **kwargs):
        self.sa_temperature = kwargs.get('sa_temperature', 100.0)
        self.sa_cooling_rate = kwargs.get('sa_cooling_rate', 0.95)
        self.sa_iterations = kwargs.get('sa_iterations', 50)
        super().__init__(genes, **kwargs)
    
    def simulated_annealing_improvement(self, temperature):
        """Apply simulated annealing for local improvement."""
        current = self
        current_fitness = current.fitness
        best = current
        best_fitness = current_fitness
        
        temp = temperature
        
        for iteration in range(self.sa_iterations):
            # Generate neighbor
            neighbor = self.get_random_neighbor()
            neighbor_fitness = neighbor.fitness
            
            # Acceptance criteria
            if neighbor_fitness > current_fitness:
                # Better solution - always accept
                current = neighbor
                current_fitness = neighbor_fitness
                if neighbor_fitness > best_fitness:
                    best = neighbor
                    best_fitness = neighbor_fitness
            else:
                # Worse solution - accept with probability
                delta = neighbor_fitness - current_fitness
                probability = np.exp(delta / temp) if temp > 0 else 0
                if random.random() < probability:
                    current = neighbor
                    current_fitness = neighbor_fitness
            
            # Cool down
            temp *= self.sa_cooling_rate
        
        return best

# Usage example
def hybrid_ga_sa_example():
    """Example combining GA with Simulated Annealing."""
    
    def complex_function(x):
        """Complex multimodal function."""
        return -(sum(x[i]**2 + 10 * np.cos(2 * np.pi * x[i]) for i in range(len(x))))
    
    class ComplexOptimizationIndividual(SAHybridIndividual):
        def generate_random_genes(self):
            return [random.uniform(-5, 5) for _ in range(5)]
        
        def calculate_fitness(self):
            return complex_function(self.genes)
        
        def get_random_neighbor(self):
            new_genes = self.genes.copy()
            idx = random.randint(0, len(new_genes) - 1)
            new_genes[idx] += random.gauss(0, 0.5)
            new_genes[idx] = max(-5, min(5, new_genes[idx]))  # Bounds
            return ComplexOptimizationIndividual(new_genes)
    
    # Run hybrid algorithm
    ga = GeneticAlgorithm(
        individual_class=ComplexOptimizationIndividual,
        population_size=50,
        generations=100,
        sa_temperature=50.0,
        sa_cooling_rate=0.9,
        sa_iterations=20
    )
    
    best, stats = ga.evolve()
    print(f"Hybrid GA-SA result: {best.fitness:.4f}")
    return best

hybrid_result = hybrid_ga_sa_example()
```

## Multi-Objective Optimization

Optimizing multiple conflicting objectives simultaneously.

### NSGA-II Implementation

```python
class MultiObjectiveIndividual(Individual):
    """Individual for multi-objective optimization."""
    
    def __init__(self, genes=None, **kwargs):
        super().__init__(genes, **kwargs)
        self.objectives = []
        self.dominance_count = 0
        self.dominated_solutions = []
        self.rank = 0
        self.crowding_distance = 0
    
    def calculate_objectives(self):
        """Calculate multiple objective values."""
        # Override in subclass to define objectives
        raise NotImplementedError
    
    def dominates(self, other):
        """Check if this individual dominates another."""
        better_in_at_least_one = False
        
        for i in range(len(self.objectives)):
            if self.objectives[i] < other.objectives[i]:
                return False  # self is worse in objective i
            elif self.objectives[i] > other.objectives[i]:
                better_in_at_least_one = True
        
        return better_in_at_least_one

class NSGAII:
    """Non-dominated Sorting Genetic Algorithm II."""
    
    def __init__(self, population_size, individual_class, **kwargs):
        self.population_size = population_size
        self.individual_class = individual_class
        self.kwargs = kwargs
        self.population = []
    
    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            individual = self.individual_class(**self.kwargs)
            individual.calculate_objectives()
            self.population.append(individual)
    
    def fast_non_dominated_sort(self, population):
        """Fast non-dominated sorting."""
        fronts = [[]]
        
        for p in population:
            p.dominance_count = 0
            p.dominated_solutions = []
            
            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.dominance_count += 1
            
            if p.dominance_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.dominance_count -= 1
                    if q.dominance_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            
            if next_front:
                fronts.append(next_front)
            i += 1
        
        return fronts
    
    def calculate_crowding_distance(self, front):
        """Calculate crowding distance for diversity preservation."""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for individual in front:
            individual.crowding_distance = 0
        
        # For each objective
        n_objectives = len(front[0].objectives)
        for obj_idx in range(n_objectives):
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            # Set boundary points to infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distances for intermediate points
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i+1].objectives[obj_idx] - 
                              front[i-1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance
    
    def tournament_selection_nsga(self, population):
        """Tournament selection based on rank and crowding distance."""
        tournament_size = 2
        selected = []
        
        for _ in range(len(population)):
            candidates = random.sample(population, tournament_size)
            
            # Select based on rank first, then crowding distance
            winner = candidates[0]
            for candidate in candidates[1:]:
                if (candidate.rank < winner.rank or 
                    (candidate.rank == winner.rank and 
                     candidate.crowding_distance > winner.crowding_distance)):
                    winner = candidate
            
            selected.append(winner)
        
        return selected
    
    def evolve(self, generations):
        """Run NSGA-II algorithm."""
        self.initialize_population()
        
        for generation in range(generations):
            # Create offspring through crossover and mutation
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(self.population, 2)
                child1, child2 = parent1.crossover(parent2)
                child1 = child1.mutate(0.1)
                child2 = child2.mutate(0.1)
                child1.calculate_objectives()
                child2.calculate_objectives()
                offspring.extend([child1, child2])
            
            # Combine parent and offspring populations
            combined = self.population + offspring
            
            # Fast non-dominated sort
            fronts = self.fast_non_dominated_sort(combined)
            
            # Select next generation
            new_population = []
            front_idx = 0
            
            while len(new_population) + len(fronts[front_idx]) <= self.population_size:
                self.calculate_crowding_distance(fronts[front_idx])
                new_population.extend(fronts[front_idx])
                front_idx += 1
            
            # Fill remaining slots from next front
            if len(new_population) < self.population_size:
                self.calculate_crowding_distance(fronts[front_idx])
                fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining_slots = self.population_size - len(new_population)
                new_population.extend(fronts[front_idx][:remaining_slots])
            
            self.population = new_population
            
            if generation % 10 == 0:
                print(f"Generation {generation}: {len(fronts)} fronts")
        
        return self.population

# Example: Multi-objective function optimization
class MultiObjectiveTestIndividual(MultiObjectiveIndividual):
    """Test individual for multi-objective optimization."""
    
    def generate_random_genes(self):
        # 2D problem
        return [random.uniform(-5, 5) for _ in range(2)]
    
    def calculate_objectives(self):
        """ZDT1 test problem."""
        x = self.genes
        
        # Objective 1: f1(x) = x1
        f1 = x[0] if x[0] >= 0 else 0
        
        # Objective 2: f2(x) = g(x) * h(f1, g(x))
        g = 1 + 9 * sum(x[1:]) / (len(x) - 1) if len(x) > 1 else 1
        h = 1 - np.sqrt(f1 / g) if g > 0 else 0
        f2 = g * h
        
        self.objectives = [f1, f2]
        # For single fitness compatibility
        self.fitness = f1 + f2  # Simple aggregation

def multi_objective_example():
    """Example of multi-objective optimization."""
    
    nsga = NSGAII(
        population_size=100,
        individual_class=MultiObjectiveTestIndividual
    )
    
    final_population = nsga.evolve(generations=100)
    
    # Extract Pareto front
    fronts = nsga.fast_non_dominated_sort(final_population)
    pareto_front = fronts[0]
    
    print(f"\nPareto front contains {len(pareto_front)} solutions")
    
    # Plot Pareto front
    import matplotlib.pyplot as plt
    
    objectives_1 = [ind.objectives[0] for ind in pareto_front]
    objectives_2 = [ind.objectives[1] for ind in pareto_front]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(objectives_1, objectives_2, c='red', alpha=0.7)
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Front')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pareto_front

pareto_solutions = multi_objective_example()
```

## Parallel and Distributed GAs

### Island Model GA

```python
import threading
import queue
import time

class IslandGA:
    """Island model for parallel genetic algorithms."""
    
    def __init__(self, n_islands, migration_rate=0.1, migration_interval=10):
        self.n_islands = n_islands
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.islands = []
        self.migration_queues = []
        self.best_overall = None
    
    def initialize_islands(self, individual_class, population_size, **kwargs):
        """Initialize all islands with random populations."""
        self.islands = []
        self.migration_queues = []
        
        for i in range(self.n_islands):
            ga = GeneticAlgorithm(
                individual_class=individual_class,
                population_size=population_size,
                generations=1,  # We'll control generations manually
                **kwargs
            )
            ga.initialize_population()
            self.islands.append(ga)
            self.migration_queues.append(queue.Queue())
    
    def migrate_individuals(self):
        """Perform migration between islands."""
        for i, island in enumerate(self.islands):
            # Send best individuals to other islands
            n_emigrants = int(len(island.population.individuals) * self.migration_rate)
            if n_emigrants > 0:
                # Select best individuals to migrate
                sorted_pop = sorted(island.population.individuals, 
                                  key=lambda x: x.fitness, reverse=True)
                emigrants = sorted_pop[:n_emigrants]
                
                # Send to random other islands
                for emigrant in emigrants:
                    target_island = random.choice(
                        [j for j in range(self.n_islands) if j != i]
                    )
                    self.migration_queues[target_island].put(emigrant)
        
        # Receive immigrants
        for i, island in enumerate(self.islands):
            immigrants = []
            while not self.migration_queues[i].empty():
                try:
                    immigrant = self.migration_queues[i].get_nowait()
                    immigrants.append(immigrant)
                except queue.Empty:
                    break
            
            if immigrants:
                # Replace worst individuals with immigrants
                sorted_pop = sorted(island.population.individuals, 
                                  key=lambda x: x.fitness)
                n_replace = min(len(immigrants), len(sorted_pop))
                
                for j in range(n_replace):
                    sorted_pop[j] = immigrants[j]
                
                island.population.individuals = sorted_pop
    
    def evolve_parallel(self, generations):
        """Evolve all islands in parallel."""
        
        def evolve_island(island_idx, island, results):
            """Evolution function for a single island."""
            for gen in range(generations):
                island.evolve_generation()
                
                # Migration every migration_interval generations
                if gen % self.migration_interval == 0 and gen > 0:
                    # Migration is handled by main thread
                    pass
            
            best = max(island.population.individuals, key=lambda x: x.fitness)
            results[island_idx] = best
        
        # Run islands in parallel
        threads = []
        results = {}
        
        for i, island in enumerate(self.islands):
            thread = threading.Thread(
                target=evolve_island, 
                args=(i, island, results)
            )
            threads.append(thread)
            thread.start()
        
        # Migration in main thread
        for gen in range(generations):
            time.sleep(0.1)  # Small delay
            if gen % self.migration_interval == 0 and gen > 0:
                self.migrate_individuals()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Find overall best
        best_solutions = list(results.values())
        self.best_overall = max(best_solutions, key=lambda x: x.fitness)
        
        return self.best_overall, results

# Usage example
def island_model_example():
    """Example using island model GA."""
    
    from genetic_algorithms.examples.tsp import TSPProblem, TSPIndividual
    
    problem = TSPProblem.create_random_problem(15, seed=42)
    
    island_ga = IslandGA(
        n_islands=4,
        migration_rate=0.1,
        migration_interval=20
    )
    
    island_ga.initialize_islands(
        individual_class=TSPIndividual,
        population_size=50,
        problem_data={'distance_matrix': problem.distance_matrix}
    )
    
    print("Running island model GA...")
    best_overall, island_results = island_ga.evolve_parallel(generations=100)
    
    print(f"\nIsland Model Results:")
    print(f"Best overall fitness: {best_overall.fitness:.2f}")
    
    for i, best in island_results.items():
        print(f"Island {i} best fitness: {best.fitness:.2f}")
    
    return best_overall

island_result = island_model_example()
```

## Adaptive Parameter Control

### Self-Adaptive Genetic Algorithm

```python
class SelfAdaptiveIndividual(Individual):
    """Individual that adapts its own parameters."""
    
    def __init__(self, genes=None, **kwargs):
        super().__init__(genes, **kwargs)
        
        # Self-adaptive parameters
        self.mutation_rate = kwargs.get('initial_mutation_rate', 0.1)
        self.mutation_step_size = kwargs.get('initial_step_size', 1.0)
        
        # Parameter bounds
        self.min_mutation_rate = 0.001
        self.max_mutation_rate = 0.5
        self.min_step_size = 0.01
        self.max_step_size = 10.0
    
    def mutate_parameters(self):
        """Mutate the strategy parameters."""
        # Mutate mutation rate
        self.mutation_rate *= np.exp(np.random.normal(0, 0.1))
        self.mutation_rate = max(self.min_mutation_rate, 
                               min(self.max_mutation_rate, self.mutation_rate))
        
        # Mutate step size
        self.mutation_step_size *= np.exp(np.random.normal(0, 0.1))
        self.mutation_step_size = max(self.min_step_size,
                                    min(self.max_step_size, self.mutation_step_size))
    
    def mutate(self, external_mutation_rate=None):
        """Mutate using self-adaptive parameters."""
        # First mutate the parameters themselves
        self.mutate_parameters()
        
        # Then mutate the solution
        new_genes = []
        for gene in self.genes:
            if random.random() < self.mutation_rate:
                if isinstance(gene, (int, float)):
                    mutation = np.random.normal(0, self.mutation_step_size)
                    new_genes.append(gene + mutation)
                else:
                    new_genes.append(gene)  # Handle non-numeric genes
            else:
                new_genes.append(gene)
        
        child = self.__class__(new_genes, **self.kwargs)
        child.mutation_rate = self.mutation_rate
        child.mutation_step_size = self.mutation_step_size
        
        return child

class AdaptiveGA(GeneticAlgorithm):
    """GA with adaptive parameter control."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Track parameter statistics
        self.mutation_rate_history = []
        self.step_size_history = []
        self.diversity_history = []
        
        # Adaptive control parameters
        self.adaptation_frequency = kwargs.get('adaptation_frequency', 10)
        self.target_diversity = kwargs.get('target_diversity', 0.5)
    
    def adapt_population_parameters(self):
        """Adapt population-level parameters based on performance."""
        if len(self.population.individuals) == 0:
            return
        
        # Calculate current diversity
        current_diversity = self.calculate_population_diversity()
        self.diversity_history.append(current_diversity)
        
        # Collect parameter statistics
        mutation_rates = []
        step_sizes = []
        
        for individual in self.population.individuals:
            if hasattr(individual, 'mutation_rate'):
                mutation_rates.append(individual.mutation_rate)
            if hasattr(individual, 'mutation_step_size'):
                step_sizes.append(individual.mutation_step_size)
        
        if mutation_rates:
            self.mutation_rate_history.append(np.mean(mutation_rates))
        if step_sizes:
            self.step_size_history.append(np.mean(step_sizes))
        
        # Adaptive parameter adjustment
        if len(self.diversity_history) >= 2:
            diversity_trend = self.diversity_history[-1] - self.diversity_history[-2]
            
            # If diversity is decreasing too fast, increase mutation
            if current_diversity < self.target_diversity * 0.5:
                self.increase_mutation_tendency()
            # If diversity is too high, decrease mutation
            elif current_diversity > self.target_diversity * 1.5:
                self.decrease_mutation_tendency()
    
    def increase_mutation_tendency(self):
        """Increase mutation rates in the population."""
        for individual in self.population.individuals:
            if hasattr(individual, 'mutation_rate'):
                individual.mutation_rate = min(0.5, individual.mutation_rate * 1.1)
    
    def decrease_mutation_tendency(self):
        """Decrease mutation rates in the population."""
        for individual in self.population.individuals:
            if hasattr(individual, 'mutation_rate'):
                individual.mutation_rate = max(0.001, individual.mutation_rate * 0.9)
    
    def calculate_population_diversity(self):
        """Calculate population diversity metric."""
        if len(self.population.individuals) < 2:
            return 0.0
        
        # Simple diversity measure: average pairwise distance
        total_distance = 0
        comparisons = 0
        
        for i in range(len(self.population.individuals)):
            for j in range(i + 1, len(self.population.individuals)):
                # Calculate distance between individuals
                ind1 = self.population.individuals[i]
                ind2 = self.population.individuals[j]
                
                if hasattr(ind1, 'genes') and hasattr(ind2, 'genes'):
                    distance = self.calculate_individual_distance(ind1, ind2)
                    total_distance += distance
                    comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def calculate_individual_distance(self, ind1, ind2):
        """Calculate distance between two individuals."""
        if len(ind1.genes) != len(ind2.genes):
            return 0.0
        
        distance = 0.0
        for g1, g2 in zip(ind1.genes, ind2.genes):
            if isinstance(g1, (int, float)) and isinstance(g2, (int, float)):
                distance += (g1 - g2) ** 2
            else:
                distance += 0 if g1 == g2 else 1
        
        return np.sqrt(distance)
    
    def evolve_generation(self):
        """Override to include adaptive parameter control."""
        # Standard evolution
        super().evolve_generation()
        
        # Adaptive parameter control
        if self.current_generation % self.adaptation_frequency == 0:
            self.adapt_population_parameters()

# Usage example
def adaptive_ga_example():
    """Example using self-adaptive GA."""
    
    class AdaptiveRealIndividual(SelfAdaptiveIndividual):
        def generate_random_genes(self):
            return [random.uniform(-10, 10) for _ in range(5)]
        
        def calculate_fitness(self):
            # Rosenbrock function
            return -sum(100 * (self.genes[i+1] - self.genes[i]**2)**2 + 
                       (1 - self.genes[i])**2 
                       for i in range(len(self.genes) - 1))
    
    adaptive_ga = AdaptiveGA(
        individual_class=AdaptiveRealIndividual,
        population_size=50,
        generations=200,
        initial_mutation_rate=0.1,
        adaptation_frequency=20,
        target_diversity=2.0
    )
    
    print("Running adaptive GA...")
    best, stats = adaptive_ga.evolve()
    
    print(f"Best fitness: {best.fitness:.6f}")
    print(f"Final mutation rate: {best.mutation_rate:.4f}")
    print(f"Final step size: {best.mutation_step_size:.4f}")
    
    # Plot parameter adaptation
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(adaptive_ga.mutation_rate_history)
    plt.title('Mutation Rate Evolution')
    plt.xlabel('Adaptation Step')
    plt.ylabel('Average Mutation Rate')
    
    plt.subplot(1, 3, 2)
    plt.plot(adaptive_ga.step_size_history)
    plt.title('Step Size Evolution')
    plt.xlabel('Adaptation Step')
    plt.ylabel('Average Step Size')
    
    plt.subplot(1, 3, 3)
    plt.plot(adaptive_ga.diversity_history)
    plt.axhline(y=adaptive_ga.target_diversity, color='r', linestyle='--', label='Target')
    plt.title('Population Diversity')
    plt.xlabel('Adaptation Step')
    plt.ylabel('Diversity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return best

adaptive_result = adaptive_ga_example()
```

## Advanced Selection Strategies

### Fitness Sharing and Niching

```python
class NichingGA(GeneticAlgorithm):
    """GA with fitness sharing for multimodal optimization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sharing_radius = kwargs.get('sharing_radius', 1.0)
        self.alpha = kwargs.get('alpha', 2.0)  # Sharing function parameter
    
    def calculate_shared_fitness(self):
        """Calculate shared fitness for all individuals."""
        n = len(self.population.individuals)
        
        for i, individual in enumerate(self.population.individuals):
            niche_count = 0
            
            for j, other in enumerate(self.population.individuals):
                distance = self.calculate_distance(individual, other)
                if distance < self.sharing_radius:
                    sharing_value = 1 - (distance / self.sharing_radius) ** self.alpha
                    niche_count += sharing_value
            
            # Adjust fitness based on niche count
            individual.shared_fitness = individual.fitness / max(niche_count, 1)
    
    def calculate_distance(self, ind1, ind2):
        """Calculate distance between individuals in genotype space."""
        if len(ind1.genes) != len(ind2.genes):
            return float('inf')
        
        distance = 0
        for g1, g2 in zip(ind1.genes, ind2.genes):
            if isinstance(g1, (int, float)) and isinstance(g2, (int, float)):
                distance += (g1 - g2) ** 2
            else:
                distance += 0 if g1 == g2 else 1
        
        return np.sqrt(distance)
    
    def selection(self):
        """Selection based on shared fitness."""
        self.calculate_shared_fitness()
        
        # Use shared fitness for selection
        selected = []
        for _ in range(self.population_size):
            if self.selection_method == 'tournament':
                candidates = random.sample(self.population.individuals, self.tournament_size)
                winner = max(candidates, key=lambda x: getattr(x, 'shared_fitness', x.fitness))
                selected.append(winner)
        
        return selected

# Usage for multimodal function optimization
def niching_ga_example():
    """Example using niching GA for multimodal optimization."""
    
    def multimodal_function(x):
        """Function with multiple peaks."""
        return np.exp(-0.5 * ((x[0] - 2)**2 + (x[1] - 2)**2)) + \
               np.exp(-0.5 * ((x[0] + 2)**2 + (x[1] + 2)**2)) + \
               0.5 * np.exp(-0.5 * ((x[0])**2 + (x[1])**2))
    
    class MultimodalIndividual(Individual):
        def generate_random_genes(self):
            return [random.uniform(-5, 5) for _ in range(2)]
        
        def calculate_fitness(self):
            return multimodal_function(self.genes)
    
    niching_ga = NichingGA(
        individual_class=MultimodalIndividual,
        population_size=100,
        generations=150,
        sharing_radius=2.0,
        alpha=2.0
    )
    
    best, stats = niching_ga.evolve()
    
    # Find all peaks (local optima)
    final_population = niching_ga.population.individuals
    peaks = []
    
    for individual in final_population:
        is_peak = True
        for other in final_population:
            if (other != individual and 
                niching_ga.calculate_distance(individual, other) < 1.0 and
                other.fitness > individual.fitness):
                is_peak = False
                break
        
        if is_peak and individual.fitness > 0.5:  # Threshold for significant peaks
            peaks.append(individual)
    
    print(f"Found {len(peaks)} peaks:")
    for i, peak in enumerate(peaks):
        print(f"Peak {i+1}: {peak.genes}, fitness: {peak.fitness:.4f}")
    
    return peaks

peaks = niching_ga_example()
```

## Constraint Handling Techniques

### Advanced Constraint Handling

```python
class ConstraintHandlingGA(GeneticAlgorithm):
    """GA with advanced constraint handling techniques."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.constraint_functions = kwargs.get('constraint_functions', [])
        self.penalty_method = kwargs.get('penalty_method', 'adaptive')
        self.penalty_factor = kwargs.get('penalty_factor', 1000.0)
        self.feasible_solutions = []
    
    def evaluate_constraints(self, individual):
        """Evaluate all constraints for an individual."""
        violations = []
        total_violation = 0
        
        for constraint_func in self.constraint_functions:
            violation = constraint_func(individual.genes)
            violations.append(max(0, violation))  # Only positive violations count
            total_violation += max(0, violation)
        
        individual.constraint_violations = violations
        individual.total_violation = total_violation
        individual.is_feasible = total_violation == 0
        
        return individual.is_feasible
    
    def calculate_penalized_fitness(self, individual):
        """Calculate fitness with constraint penalties."""
        base_fitness = individual.fitness
        
        if individual.is_feasible:
            return base_fitness
        
        if self.penalty_method == 'static':
            penalty = self.penalty_factor * individual.total_violation
        elif self.penalty_method == 'adaptive':
            # Adaptive penalty increases over generations
            penalty = (self.penalty_factor * 
                      (1 + self.current_generation / self.generations) * 
                      individual.total_violation)
        elif self.penalty_method == 'dynamic':
            # Dynamic penalty based on feasible solutions ratio
            feasible_ratio = len(self.feasible_solutions) / len(self.population.individuals)
            penalty = self.penalty_factor * (1 + feasible_ratio) * individual.total_violation
        
        return base_fitness - penalty
    
    def tournament_selection_constrained(self):
        """Tournament selection with constraint handling."""
        selected = []
        
        for _ in range(self.population_size):
            candidates = random.sample(self.population.individuals, self.tournament_size)
            
            # Prefer feasible solutions
            feasible_candidates = [c for c in candidates if c.is_feasible]
            
            if feasible_candidates:
                # Tournament among feasible solutions
                winner = max(feasible_candidates, key=lambda x: x.fitness)
            else:
                # Tournament among infeasible solutions (least violation)
                winner = min(candidates, key=lambda x: x.total_violation)
            
            selected.append(winner)
        
        return selected
    
    def evolve_generation(self):
        """Override to include constraint evaluation."""
        # Evaluate constraints
        self.feasible_solutions = []
        for individual in self.population.individuals:
            self.evaluate_constraints(individual)
            if individual.is_feasible:
                self.feasible_solutions.append(individual)
        
        # Calculate penalized fitness
        for individual in self.population.individuals:
            individual.penalized_fitness = self.calculate_penalized_fitness(individual)
        
        # Use constraint-aware selection
        selected = self.tournament_selection_constrained()
        
        # Rest of evolution process
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            if random.random() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1, parent2
            
            child1 = child1.mutate(self.mutation_rate)
            child2 = child2.mutate(self.mutation_rate)
            
            offspring.extend([child1, child2])
        
        # Replace population
        self.population.individuals = offspring[:self.population_size]
        self.population.evaluate_fitness()

# Example: Constrained optimization problem
def constrained_optimization_example():
    """Example with multiple constraints."""
    
    def objective_function(x):
        """Minimize: (x1-2)^2 + (x2-1)^2"""
        return -((x[0] - 2)**2 + (x[1] - 1)**2)
    
    def constraint1(x):
        """g1: x1^2 + x2^2 - 1 <= 0 (inside unit circle)"""
        return x[0]**2 + x[1]**2 - 1
    
    def constraint2(x):
        """g2: x1 + x2 - 1 <= 0"""
        return x[0] + x[1] - 1
    
    class ConstrainedIndividual(Individual):
        def generate_random_genes(self):
            return [random.uniform(-2, 2) for _ in range(2)]
        
        def calculate_fitness(self):
            return objective_function(self.genes)
    
    constrained_ga = ConstraintHandlingGA(
        individual_class=ConstrainedIndividual,
        population_size=100,
        generations=200,
        constraint_functions=[constraint1, constraint2],
        penalty_method='adaptive',
        penalty_factor=1000.0
    )
    
    best, stats = constrained_ga.evolve()
    
    print(f"Best solution: x1={best.genes[0]:.4f}, x2={best.genes[1]:.4f}")
    print(f"Objective value: {-best.fitness:.4f}")
    print(f"Is feasible: {best.is_feasible}")
    print(f"Constraint violations: {best.constraint_violations}")
    
    # Check final population feasibility
    feasible_count = sum(1 for ind in constrained_ga.population.individuals if ind.is_feasible)
    print(f"Feasible solutions in final population: {feasible_count}/{len(constrained_ga.population.individuals)}")
    
    return best

constrained_result = constrained_optimization_example()
```

## Real-World Deployment Considerations

### Production-Ready GA Implementation

```python
import logging
import json
import pickle
from datetime import datetime
from typing import Optional, Dict, Any, List
import os

class ProductionGA:
    """Production-ready GA with logging, checkpointing, and monitoring."""
    
    def __init__(self, config_file: Optional[str] = None, **kwargs):
        # Load configuration
        self.config = self.load_config(config_file, kwargs)
        
        # Set up logging
        self.setup_logging()
        
        # Initialize components
        self.ga = None
        self.checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        self.results_dir = self.config.get('results_dir', './results')
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Performance monitoring
        self.start_time = None
        self.generation_times = []
        self.memory_usage = []
    
    def load_config(self, config_file: Optional[str], kwargs: Dict) -> Dict:
        """Load configuration from file or kwargs."""
        config = {}
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Override with kwargs
        config.update(kwargs)
        
        return config
    
    def setup_logging(self):
        """Set up comprehensive logging."""
        log_level = self.config.get('log_level', 'INFO')
        log_file = self.config.get('log_file', f'ga_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, generation: int, population, stats_history):
        """Save algorithm state for recovery."""
        checkpoint_data = {
            'generation': generation,
            'population': population,
            'stats_history': stats_history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_gen_{generation}.pkl'
        )
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Checkpoint saved at generation {generation}")
            
            # Keep only last N checkpoints
            self.cleanup_old_checkpoints()
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """Load algorithm state from checkpoint."""
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_file}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoint files."""
        try:
            checkpoint_files = [
                f for f in os.listdir(self.checkpoint_dir) 
                if f.startswith('checkpoint_gen_') and f.endswith('.pkl')
            ]
            
            # Sort by generation number
            checkpoint_files.sort(
                key=lambda x: int(x.split('_')[2].split('.')[0])
            )
            
            # Remove old files
            files_to_remove = checkpoint_files[:-keep_last]
            for file_to_remove in files_to_remove:
                os.remove(os.path.join(self.checkpoint_dir, file_to_remove))
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup checkpoints: {e}")
    
    def monitor_performance(self, generation: int):
        """Monitor algorithm performance."""
        import psutil
        
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        
        # Log performance metrics
        if generation % 10 == 0:
            avg_gen_time = np.mean(self.generation_times[-10:]) if self.generation_times else 0
            self.logger.info(
                f"Gen {generation}: Avg time/gen: {avg_gen_time:.2f}s, "
                f"Memory: {memory_mb:.1f}MB"
            )
    
    def run_optimization(self, 
                        individual_class,
                        resume_from_checkpoint: Optional[str] = None) -> Dict:
        """Run the complete optimization with monitoring."""
        
        self.start_time = datetime.now()
        self.logger.info("Starting genetic algorithm optimization")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        try:
            # Initialize or resume
            if resume_from_checkpoint:
                checkpoint_data = self.load_checkpoint(resume_from_checkpoint)
                if checkpoint_data:
                    self.ga = GeneticAlgorithm(**self.config)
                    self.ga.population = checkpoint_data['population']
                    stats_history = checkpoint_data['stats_history']
                    start_generation = checkpoint_data['generation']
                else:
                    raise ValueError("Failed to load checkpoint")
            else:
                self.ga = GeneticAlgorithm(
                    individual_class=individual_class,
                    **self.config
                )
                stats_history = []
                start_generation = 0
            
            # Run evolution with monitoring
            total_generations = self.config.get('generations', 100)
            checkpoint_interval = self.config.get('checkpoint_interval', 50)
            
            for generation in range(start_generation, total_generations):
                gen_start_time = datetime.now()
                
                # Run one generation
                if generation == 0 and not resume_from_checkpoint:
                    self.ga.initialize_population()
                else:
                    self.ga.evolve_generation()
                
                # Record statistics
                stats = self.ga.get_generation_stats()
                stats_history.append(stats)
                
                # Performance monitoring
                gen_time = (datetime.now() - gen_start_time).total_seconds()
                self.generation_times.append(gen_time)
                self.monitor_performance(generation)
                
                # Checkpointing
                if generation % checkpoint_interval == 0:
                    self.save_checkpoint(generation, self.ga.population, stats_history)
                
                # Early stopping check
                if self.check_early_stopping(stats_history):
                    self.logger.info(f"Early stopping at generation {generation}")
                    break
            
            # Final results
            best_individual = max(self.ga.population.individuals, key=lambda x: x.fitness)
            
            # Save final results
            results = {
                'best_individual': best_individual,
                'stats_history': stats_history,
                'config': self.config,
                'runtime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'total_generations': len(stats_history),
                'memory_usage': self.memory_usage,
                'generation_times': self.generation_times
            }
            
            self.save_results(results)
            
            self.logger.info("Optimization completed successfully")
            self.logger.info(f"Best fitness: {best_individual.fitness}")
            self.logger.info(f"Total runtime: {results['runtime_seconds']:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def check_early_stopping(self, stats_history: List[Dict]) -> bool:
        """Check if early stopping criteria are met."""
        if len(stats_history) < 2:
            return False
        
        # Check for stagnation
        stagnation_generations = self.config.get('stagnation_generations', 50)
        target_fitness = self.config.get('target_fitness')
        
        if len(stats_history) >= stagnation_generations:
            recent_best = [s['best_fitness'] for s in stats_history[-stagnation_generations:]]
            if max(recent_best) - min(recent_best) < 1e-6:
                return True
        
        # Check if target fitness reached
        if target_fitness and stats_history[-1]['best_fitness'] >= target_fitness:
            return True
        
        return False
    
    def save_results(self, results: Dict):
        """Save final results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'ga_results_{timestamp}.json')
        
        # Convert non-serializable objects
        serializable_results = {
            'best_fitness': results['best_individual'].fitness,
            'best_genes': results['best_individual'].genes,
            'config': results['config'],
            'runtime_seconds': results['runtime_seconds'],
            'total_generations': results['total_generations'],
            'stats_summary': {
                'initial_best_fitness': results['stats_history'][0]['best_fitness'],
                'final_best_fitness': results['stats_history'][-1]['best_fitness'],
                'improvement': (results['stats_history'][-1]['best_fitness'] - 
                              results['stats_history'][0]['best_fitness'])
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")

# Usage example
def production_deployment_example():
    """Example of production-ready GA deployment."""
    
    # Configuration file
    config = {
        'population_size': 100,
        'generations': 500,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'selection_method': 'tournament',
        'tournament_size': 5,
        'elitism': True,
        'elite_size': 5,
        'checkpoint_interval': 25,
        'stagnation_generations': 100,
        'target_fitness': 0.99,
        'log_level': 'INFO'
    }
    
    from genetic_algorithms.examples.knapsack import KnapsackProblem, KnapsackIndividual
    
    # Create problem
    problem = KnapsackProblem.create_random_problem(50, capacity=100, seed=42)
    config['problem_data'] = {'items': problem.items, 'capacity': problem.capacity}
    
    # Run production GA
    production_ga = ProductionGA(**config)
    
    results = production_ga.run_optimization(
        individual_class=KnapsackIndividual
    )
    
    print(f"Production GA completed:")
    print(f"Best fitness: {results['best_individual'].fitness}")
    print(f"Runtime: {results['runtime_seconds']:.2f} seconds")
    print(f"Generations: {results['total_generations']}")
    
    return results

# Run production example
production_results = production_deployment_example()
```

This completes the advanced topics documentation, covering sophisticated genetic algorithm techniques for real-world applications.