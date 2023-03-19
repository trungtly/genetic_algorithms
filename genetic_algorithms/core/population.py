from typing import List, Type, Callable, Optional, Tuple
import random
import statistics
from .individual import Individual


class Population:
    """
    Manages a population of individuals in a genetic algorithm.
    
    The population is the collection of all candidate solutions (individuals)
    that evolve over generations. This class provides methods for:
    - Population initialization
    - Statistical analysis
    - Population updates
    - Elitism management
    
    Key concepts:
    - Population size: Number of individuals in each generation
    - Diversity: How different the individuals are from each other
    - Convergence: When the population becomes too similar
    """
    
    def __init__(self, 
                 individual_class: Type[Individual],
                 size: int,
                 **individual_kwargs):
        """
        Initialize a population.
        
        Args:
            individual_class: Class to use for creating individuals
            size: Number of individuals in the population
            **individual_kwargs: Additional arguments for individual creation
        """
        self.individual_class = individual_class
        self.size = size
        self.individual_kwargs = individual_kwargs
        self.individuals: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.fitness_history: List[float] = []
        
        # Initialize the population with random individuals
        self._initialize_population()
    
    def _initialize_population(self):
        """
        Create initial population with random individuals.
        
        Each individual is created using the specified individual_class
        with random genes.
        """
        self.individuals = []
        for _ in range(self.size):
            individual = self.individual_class(**self.individual_kwargs)
            self.individuals.append(individual)
        
        # Calculate initial fitness values
        self._evaluate_population()
    
    def _evaluate_population(self):
        """
        Calculate fitness for all individuals in the population.
        
        This method also updates population statistics and tracks
        the best individual ever found.
        """
        for individual in self.individuals:
            individual.get_fitness()  # This calculates fitness if needed
        
        # Update best ever individual
        current_best = self.get_best_individual()
        if self.best_ever is None or current_best.get_fitness() > self.best_ever.get_fitness():
            self.best_ever = current_best.copy()
        
        # Record fitness statistics
        self.fitness_history.append(self.get_average_fitness())
    
    def get_best_individual(self) -> Individual:
        """
        Get the individual with the highest fitness in current population.
        
        Returns:
            Individual with maximum fitness
        """
        return max(self.individuals, key=lambda ind: ind.get_fitness())
    
    def get_worst_individual(self) -> Individual:
        """
        Get the individual with the lowest fitness in current population.
        
        Returns:
            Individual with minimum fitness
        """
        return min(self.individuals, key=lambda ind: ind.get_fitness())
    
    def get_average_fitness(self) -> float:
        """
        Calculate the average fitness of the population.
        
        Returns:
            Mean fitness value
        """
        if not self.individuals:
            return 0.0
        return statistics.mean(ind.get_fitness() for ind in self.individuals)
    
    def get_fitness_std(self) -> float:
        """
        Calculate the standard deviation of fitness values.
        
        Standard deviation measures population diversity:
        - High std: diverse population
        - Low std: converged population
        
        Returns:
            Standard deviation of fitness values
        """
        if len(self.individuals) < 2:
            return 0.0
        return statistics.stdev(ind.get_fitness() for ind in self.individuals)
    
    def get_fitness_stats(self) -> dict:
        """
        Get comprehensive fitness statistics for the population.
        
        Returns:
            Dictionary containing min, max, mean, and std of fitness values
        """
        fitness_values = [ind.get_fitness() for ind in self.individuals]
        return {
            'min': min(fitness_values),
            'max': max(fitness_values),
            'mean': statistics.mean(fitness_values),
            'std': statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0.0,
            'median': statistics.median(fitness_values)
        }
    
    def sort_by_fitness(self, reverse: bool = True):
        """
        Sort the population by fitness.
        
        Args:
            reverse: If True, sort in descending order (best first)
        """
        self.individuals.sort(key=lambda ind: ind.get_fitness(), reverse=reverse)
    
    def replace_population(self, new_individuals: List[Individual]):
        """
        Replace the current population with a new set of individuals.
        
        Args:
            new_individuals: List of new individuals to form the population
        """
        if len(new_individuals) != self.size:
            raise ValueError(f"New population size {len(new_individuals)} "
                           f"doesn't match expected size {self.size}")
        
        self.individuals = new_individuals
        self.generation += 1
        
        # Age all individuals
        for individual in self.individuals:
            individual.age_increment()
        
        # Update statistics
        self._evaluate_population()
    
    def select_individuals(self, count: int, method: str = 'tournament') -> List[Individual]:
        """
        Select individuals from the population for reproduction.
        
        Args:
            count: Number of individuals to select
            method: Selection method ('tournament', 'roulette', 'rank', 'random')
            
        Returns:
            List of selected individuals
        """
        if method == 'tournament':
            return self._tournament_selection(count)
        elif method == 'roulette':
            return self._roulette_selection(count)
        elif method == 'rank':
            return self._rank_selection(count)
        elif method == 'random':
            return self._random_selection(count)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _tournament_selection(self, count: int, tournament_size: int = 3) -> List[Individual]:
        """
        Tournament selection: randomly select tournament_size individuals
        and choose the best one. Repeat count times.
        
        Args:
            count: Number of individuals to select
            tournament_size: Size of each tournament
            
        Returns:
            List of selected individuals
        """
        selected = []
        for _ in range(count):
            # Randomly select tournament participants
            tournament = random.sample(self.individuals, tournament_size)
            # Choose the best from the tournament
            winner = max(tournament, key=lambda ind: ind.get_fitness())
            selected.append(winner)
        return selected
    
    def _roulette_selection(self, count: int) -> List[Individual]:
        """
        Roulette wheel selection: probability of selection proportional to fitness.
        
        Args:
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        # Calculate total fitness (handle negative fitness values)
        fitness_values = [ind.get_fitness() for ind in self.individuals]
        min_fitness = min(fitness_values)
        
        # Shift fitness values to be non-negative
        adjusted_fitness = [f - min_fitness + 1 for f in fitness_values]
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            # If all fitness values are the same, use random selection
            return self._random_selection(count)
        
        selected = []
        for _ in range(count):
            # Spin the roulette wheel
            spin = random.uniform(0, total_fitness)
            cumulative_fitness = 0
            
            for i, individual in enumerate(self.individuals):
                cumulative_fitness += adjusted_fitness[i]
                if cumulative_fitness >= spin:
                    selected.append(individual)
                    break
        
        return selected
    
    def _rank_selection(self, count: int) -> List[Individual]:
        """
        Rank selection: probability of selection based on rank, not raw fitness.
        
        Args:
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        # Sort individuals by fitness
        sorted_individuals = sorted(self.individuals, 
                                  key=lambda ind: ind.get_fitness())
        
        # Assign ranks (1 to population_size)
        ranks = list(range(1, len(sorted_individuals) + 1))
        total_rank = sum(ranks)
        
        selected = []
        for _ in range(count):
            # Select based on rank probability
            spin = random.uniform(0, total_rank)
            cumulative_rank = 0
            
            for i, individual in enumerate(sorted_individuals):
                cumulative_rank += ranks[i]
                if cumulative_rank >= spin:
                    selected.append(individual)
                    break
        
        return selected
    
    def _random_selection(self, count: int) -> List[Individual]:
        """
        Random selection: each individual has equal probability of selection.
        
        Args:
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        return random.choices(self.individuals, k=count)
    
    def apply_elitism(self, elite_count: int) -> List[Individual]:
        """
        Select the best individuals for elitism.
        
        Elitism ensures that the best individuals from the current
        generation survive to the next generation unchanged.
        
        Args:
            elite_count: Number of elite individuals to preserve
            
        Returns:
            List of elite individuals (copies)
        """
        if elite_count <= 0:
            return []
        
        # Sort population by fitness (best first)
        self.sort_by_fitness(reverse=True)
        
        # Return copies of the best individuals
        elite = []
        for i in range(min(elite_count, len(self.individuals))):
            elite.append(self.individuals[i].copy())
        
        return elite
    
    def get_diversity_measure(self) -> float:
        """
        Calculate a measure of population diversity.
        
        This implementation uses the standard deviation of fitness values
        as a simple diversity measure. More sophisticated measures could
        consider genetic distance between individuals.
        
        Returns:
            Diversity measure (higher = more diverse)
        """
        return self.get_fitness_std()
    
    def is_converged(self, threshold: float = 1e-6) -> bool:
        """
        Check if the population has converged.
        
        A population is considered converged if the fitness variance
        is below a threshold, indicating all individuals are very similar.
        
        Args:
            threshold: Convergence threshold for fitness standard deviation
            
        Returns:
            True if population has converged
        """
        return self.get_fitness_std() < threshold
    
    def get_age_stats(self) -> dict:
        """
        Get statistics about individual ages in the population.
        
        Returns:
            Dictionary with age statistics
        """
        ages = [ind.age for ind in self.individuals]
        return {
            'min_age': min(ages),
            'max_age': max(ages),
            'mean_age': statistics.mean(ages),
            'std_age': statistics.stdev(ages) if len(ages) > 1 else 0.0
        }
    
    def __str__(self) -> str:
        """String representation of the population."""
        stats = self.get_fitness_stats()
        return (f"Population(gen={self.generation}, size={self.size}, "
                f"fitness: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                f"mean={stats['mean']:.4f}, std={stats['std']:.4f})")
    
    def __len__(self) -> int:
        """Return the size of the population."""
        return len(self.individuals)
    
    def __iter__(self):
        """Make the population iterable."""
        return iter(self.individuals)
    
    def __getitem__(self, index) -> Individual:
        """Allow indexing into the population."""
        return self.individuals[index]