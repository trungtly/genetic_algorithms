from typing import List, Optional, Dict, Any, Callable, Tuple, Type
import random
import time
import statistics
from .individual import Individual
from .population import Population
from .selection import SelectionOperator, TournamentSelection
from .crossover import CrossoverOperator, get_default_crossover
from .mutation import MutationOperator, get_default_mutation


class GeneticAlgorithm:
    """
    Main Genetic Algorithm engine that orchestrates the evolutionary process.
    
    A genetic algorithm follows this basic cycle:
    1. Initialize population with random individuals
    2. Evaluate fitness of all individuals
    3. Select parents for reproduction
    4. Create offspring through crossover and mutation
    5. Replace old population with new generation
    6. Repeat until termination criteria are met
    
    This implementation provides:
    - Flexible operator selection (selection, crossover, mutation)
    - Multiple termination criteria
    - Elitism support
    - Comprehensive statistics tracking
    - Progress callbacks for monitoring
    
    Key parameters:
    - Population size: Number of individuals per generation
    - Mutation rate: Probability of mutating each gene
    - Crossover rate: Probability of applying crossover
    - Elite count: Number of best individuals preserved each generation
    """
    
    def __init__(self,
                 population_size: int = 100,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8,
                 elite_count: int = 2,
                 selection_operator: Optional[SelectionOperator] = None,
                 crossover_operator: Optional[CrossoverOperator] = None,
                 mutation_operator: Optional[MutationOperator] = None,
                 maximize: bool = True):
        """
        Initialize the genetic algorithm.
        
        Args:
            population_size: Number of individuals in each generation
            mutation_rate: Probability of mutating each gene (0.0 to 1.0)
            crossover_rate: Probability of applying crossover (0.0 to 1.0)
            elite_count: Number of best individuals to preserve each generation
            selection_operator: Selection method (default: tournament selection)
            crossover_operator: Crossover method (auto-selected if None)
            mutation_operator: Mutation method (auto-selected if None)  
            maximize: True to maximize fitness, False to minimize
        """
        # Validate parameters
        if population_size < 2:
            raise ValueError("Population size must be at least 2")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        if elite_count < 0:
            raise ValueError("Elite count must be non-negative")
        if elite_count >= population_size:
            raise ValueError("Elite count must be less than population size")
        
        # Store parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.maximize = maximize
        
        # Operators (will be set when problem is provided)
        self.selection_operator = selection_operator or TournamentSelection()
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        
        # Evolution state
        self.population: Optional[Population] = None
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # Timing and statistics
        self.start_time: Optional[float] = None
        self.evolution_time: float = 0.0
        self.evaluations_count: int = 0
        
        # Callbacks and monitoring
        self.progress_callback: Optional[Callable[[int, float, float], None]] = None
        self.generation_callback: Optional[Callable[['GeneticAlgorithm'], None]] = None
    
    def evolve(self,
               individual_class: Type[Individual],
               max_generations: int = 1000,
               target_fitness: Optional[float] = None,
               max_time: Optional[float] = None,
               max_evaluations: Optional[int] = None,
               convergence_threshold: float = 1e-6,
               convergence_generations: int = 50,
               verbose: bool = True,
               **individual_kwargs) -> Individual:
        """
        Run the genetic algorithm to evolve a solution.
        
        Args:
            individual_class: Class to use for creating individuals
            max_generations: Maximum number of generations to run
            target_fitness: Stop when this fitness is reached
            max_time: Maximum runtime in seconds
            max_evaluations: Maximum number of fitness evaluations
            convergence_threshold: Stop when population diversity falls below this
            convergence_generations: Number of generations to check for convergence
            verbose: Print progress information
            **individual_kwargs: Additional arguments for individual creation
            
        Returns:
            Best individual found during evolution
        """
        # Initialize evolution
        self._initialize_evolution(individual_class, **individual_kwargs)
        
        if verbose:
            print(f"Starting evolution with {self.population_size} individuals")
            print(f"Operators: {self.selection_operator}, {self.crossover_operator}, {self.mutation_operator}")
        
        # Main evolution loop
        self.start_time = time.time()
        
        try:
            while not self._should_terminate(max_generations, target_fitness, max_time,
                                           max_evaluations, convergence_threshold,
                                           convergence_generations):
                
                # Create next generation
                self._evolve_generation()
                
                # Update statistics and history
                self._update_statistics()
                
                # Call callbacks
                if self.progress_callback:
                    self.progress_callback(self.generation, 
                                         self.best_individual.get_fitness(),
                                         self.population.get_average_fitness())
                
                if self.generation_callback:
                    self.generation_callback(self)
                
                # Print progress
                if verbose and self.generation % max(1, max_generations // 20) == 0:
                    self._print_progress()
        
        except KeyboardInterrupt:
            if verbose:
                print(f"\nEvolution interrupted at generation {self.generation}")
        
        # Finalize evolution
        self.evolution_time = time.time() - self.start_time
        
        if verbose:
            self._print_final_results()
        
        return self.best_individual
    
    def _initialize_evolution(self, individual_class: Type[Individual], **individual_kwargs):
        """Initialize the evolution process."""
        # Auto-select operators if not provided
        if self.crossover_operator is None:
            self.crossover_operator = get_default_crossover(individual_class)
        if self.mutation_operator is None:
            self.mutation_operator = get_default_mutation(individual_class)
        
        # Create initial population
        self.population = Population(individual_class, self.population_size, **individual_kwargs)
        self.generation = 0
        self.evaluations_count = self.population_size
        
        # Initialize best individual
        self.best_individual = self.population.get_best_individual().copy()
        
        # Initialize history
        self.best_fitness_history = [self.best_individual.get_fitness()]
        self.average_fitness_history = [self.population.get_average_fitness()]
        self.diversity_history = [self.population.get_diversity_measure()]
    
    def _evolve_generation(self):
        """Create the next generation through selection, crossover, and mutation."""
        new_individuals = []
        
        # Elitism: preserve best individuals
        if self.elite_count > 0:
            elite_individuals = self.population.apply_elitism(self.elite_count)
            new_individuals.extend(elite_individuals)
        
        # Generate offspring to fill remaining slots
        offspring_needed = self.population_size - len(new_individuals)
        
        while len(new_individuals) < self.population_size:
            # Select parents
            parent1, parent2 = self.selection_operator.select(self.population, 2)
            
            # Apply crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover_operator.crossover(parent1, parent2)
            else:
                # No crossover: offspring are copies of parents
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Apply mutation
            self.mutation_operator.mutate(offspring1, self.mutation_rate)
            self.mutation_operator.mutate(offspring2, self.mutation_rate)
            
            # Add offspring to new generation
            new_individuals.append(offspring1)
            if len(new_individuals) < self.population_size:
                new_individuals.append(offspring2)
        
        # Trim to exact population size if needed
        new_individuals = new_individuals[:self.population_size]
        
        # Replace population
        self.population.replace_population(new_individuals)
        self.generation += 1
        self.evaluations_count += len(new_individuals)
    
    def _update_statistics(self):
        """Update evolution statistics and best individual."""
        # Update best individual
        current_best = self.population.get_best_individual()
        if self._is_better(current_best.get_fitness(), self.best_individual.get_fitness()):
            self.best_individual = current_best.copy()
        
        # Update history
        self.best_fitness_history.append(self.best_individual.get_fitness())
        self.average_fitness_history.append(self.population.get_average_fitness())
        self.diversity_history.append(self.population.get_diversity_measure())
    
    def _should_terminate(self, max_generations: int, target_fitness: Optional[float],
                         max_time: Optional[float], max_evaluations: Optional[int],
                         convergence_threshold: float, convergence_generations: int) -> bool:
        """Check if any termination criteria are met."""
        
        # Maximum generations
        if self.generation >= max_generations:
            return True
        
        # Target fitness reached
        if target_fitness is not None:
            if self.maximize and self.best_individual.get_fitness() >= target_fitness:
                return True
            elif not self.maximize and self.best_individual.get_fitness() <= target_fitness:
                return True
        
        # Maximum time exceeded
        if max_time is not None and self.start_time is not None:
            if time.time() - self.start_time >= max_time:
                return True
        
        # Maximum evaluations exceeded
        if max_evaluations is not None and self.evaluations_count >= max_evaluations:
            return True
        
        # Population convergence
        if (len(self.diversity_history) >= convergence_generations and
            all(d < convergence_threshold for d in self.diversity_history[-convergence_generations:])):
            return True
        
        return False
    
    def _is_better(self, fitness1: float, fitness2: float) -> bool:
        """Compare two fitness values based on optimization direction."""
        if self.maximize:
            return fitness1 > fitness2
        else:
            return fitness1 < fitness2
    
    def _print_progress(self):
        """Print current evolution progress."""
        stats = self.population.get_fitness_stats()
        print(f"Generation {self.generation:4d}: "
              f"Best={stats['max']:.6f}, "
              f"Avg={stats['mean']:.6f}, "
              f"Std={stats['std']:.6f}")
    
    def _print_final_results(self):
        """Print final evolution results."""
        print(f"\nEvolution completed:")
        print(f"  Generations: {self.generation}")
        print(f"  Time: {self.evolution_time:.2f}s")
        print(f"  Evaluations: {self.evaluations_count}")
        print(f"  Best fitness: {self.best_individual.get_fitness():.6f}")
        print(f"  Final population diversity: {self.diversity_history[-1]:.6f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the evolution process.
        
        Returns:
            Dictionary containing evolution statistics
        """
        return {
            'generation': self.generation,
            'evolution_time': self.evolution_time,
            'evaluations_count': self.evaluations_count,
            'best_fitness': self.best_individual.get_fitness() if self.best_individual else None,
            'best_fitness_history': self.best_fitness_history,
            'average_fitness_history': self.average_fitness_history,
            'diversity_history': self.diversity_history,
            'final_population_stats': self.population.get_fitness_stats() if self.population else None,
            'parameters': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_count': self.elite_count,
                'maximize': self.maximize
            },
            'operators': {
                'selection': str(self.selection_operator),
                'crossover': str(self.crossover_operator),
                'mutation': str(self.mutation_operator)
            }
        }
    
    def set_progress_callback(self, callback: Callable[[int, float, float], None]):
        """
        Set a callback function to monitor progress.
        
        Args:
            callback: Function called each generation with (generation, best_fitness, avg_fitness)
        """
        self.progress_callback = callback
    
    def set_generation_callback(self, callback: Callable[['GeneticAlgorithm'], None]):
        """
        Set a callback function called after each generation.
        
        Args:
            callback: Function called with the GeneticAlgorithm instance
        """
        self.generation_callback = callback
    
    def get_best_individual(self) -> Optional[Individual]:
        """Get the best individual found so far."""
        return self.best_individual
    
    def get_population(self) -> Optional[Population]:
        """Get the current population."""
        return self.population
    
    def get_fitness_history(self) -> Tuple[List[float], List[float]]:
        """
        Get fitness history over generations.
        
        Returns:
            Tuple of (best_fitness_history, average_fitness_history)
        """
        return self.best_fitness_history, self.average_fitness_history
    
    def restart_evolution(self):
        """Reset the algorithm state for a fresh evolution run."""
        self.population = None
        self.generation = 0
        self.best_individual = None
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.diversity_history = []
        self.start_time = None
        self.evolution_time = 0.0
        self.evaluations_count = 0
    
    def __str__(self) -> str:
        """String representation of the genetic algorithm."""
        return (f"GeneticAlgorithm(pop_size={self.population_size}, "
                f"mutation_rate={self.mutation_rate}, "
                f"crossover_rate={self.crossover_rate}, "
                f"elite_count={self.elite_count})")


# Convenience function for quick problem solving

def solve_with_ga(individual_class: Type[Individual],
                  population_size: int = 100,
                  max_generations: int = 1000,
                  mutation_rate: float = 0.01,
                  crossover_rate: float = 0.8,
                  elite_count: int = 2,
                  verbose: bool = True,
                  **individual_kwargs) -> Tuple[Individual, Dict[str, Any]]:
    """
    Quick function to solve a problem using genetic algorithm with default settings.
    
    Args:
        individual_class: Class representing the problem individuals
        population_size: Size of the population
        max_generations: Maximum number of generations
        mutation_rate: Probability of gene mutation
        crossover_rate: Probability of crossover
        elite_count: Number of elite individuals to preserve
        verbose: Print progress information
        **individual_kwargs: Additional arguments for individual creation
        
    Returns:
        Tuple of (best_individual, statistics_dict)
    """
    ga = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elite_count=elite_count
    )
    
    best_individual = ga.evolve(
        individual_class=individual_class,
        max_generations=max_generations,
        verbose=verbose,
        **individual_kwargs
    )
    
    statistics = ga.get_statistics()
    
    return best_individual, statistics