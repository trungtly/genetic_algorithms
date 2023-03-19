from typing import List, Tuple, Optional
import random
import math
from .individual import Individual
from .population import Population


class SelectionOperator:
    """
    Base class for selection operators in genetic algorithms.
    
    Selection operators determine which individuals from the current
    population will be chosen to produce offspring for the next generation.
    Different selection methods apply different selection pressures:
    
    - High selection pressure: Strongly favors fit individuals
    - Low selection pressure: More random, maintains diversity
    
    The choice of selection operator significantly affects:
    - Convergence speed
    - Population diversity
    - Risk of premature convergence
    """
    
    def select(self, population: Population, count: int) -> List[Individual]:
        """
        Select individuals from a population.
        
        Args:
            population: The population to select from
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        raise NotImplementedError("Subclasses must implement select method")


class TournamentSelection(SelectionOperator):
    """
    Tournament Selection: Randomly select k individuals and choose the best.
    
    Tournament selection is one of the most popular selection methods because:
    - It's simple to implement and understand
    - It provides good selection pressure control via tournament size
    - It doesn't require fitness scaling or sorting the entire population
    - It works well with any fitness landscape
    
    Tournament size effects:
    - Larger tournaments → Higher selection pressure → Faster convergence
    - Smaller tournaments → Lower selection pressure → More diversity
    
    Typical tournament sizes: 2-7 individuals
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament.
                           Common values: 2 (binary tournament), 3, 4, 5
        """
        if tournament_size < 1:
            raise ValueError("Tournament size must be at least 1")
        self.tournament_size = tournament_size
    
    def select(self, population: Population, count: int) -> List[Individual]:
        """
        Perform tournament selection.
        
        For each selection:
        1. Randomly choose tournament_size individuals
        2. Select the individual with highest fitness
        3. Return selected individual (with replacement)
        
        Args:
            population: Population to select from
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        selected = []
        
        for _ in range(count):
            # Randomly select tournament participants
            tournament_size = min(self.tournament_size, len(population))
            tournament = random.sample(population.individuals, tournament_size)
            
            # Find the winner (individual with highest fitness)
            winner = max(tournament, key=lambda ind: ind.get_fitness())
            selected.append(winner)
        
        return selected
    
    def __str__(self) -> str:
        return f"TournamentSelection(size={self.tournament_size})"


class RouletteWheelSelection(SelectionOperator):
    """
    Roulette Wheel Selection: Fitness-proportionate selection.
    
    Also known as fitness-proportionate selection. Each individual's
    selection probability is proportional to their fitness value.
    
    Imagine a roulette wheel where each individual gets a slice
    proportional to their fitness:
    - High fitness → Large slice → High selection probability
    - Low fitness → Small slice → Low selection probability
    
    Advantages:
    - Intuitive and biologically inspired
    - All individuals have some chance of selection
    - Natural fitness scaling
    
    Disadvantages:
    - Problems with negative fitness values
    - Can be dominated by super-fit individuals
    - Sensitive to fitness scaling
    """
    
    def __init__(self, scaling_method: str = 'linear'):
        """
        Initialize roulette wheel selection.
        
        Args:
            scaling_method: Method to handle negative fitness values
                          'linear': Shift all fitness values to be positive
                          'exponential': Use exponential scaling
        """
        self.scaling_method = scaling_method
    
    def select(self, population: Population, count: int) -> List[Individual]:
        """
        Perform roulette wheel selection.
        
        Args:
            population: Population to select from
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        # Get fitness values and handle negative values
        fitness_values = [ind.get_fitness() for ind in population.individuals]
        scaled_fitness = self._scale_fitness(fitness_values)
        
        total_fitness = sum(scaled_fitness)
        
        if total_fitness == 0:
            # If all fitness values are zero, use random selection
            return random.choices(population.individuals, k=count)
        
        selected = []
        
        for _ in range(count):
            # Spin the roulette wheel
            spin = random.uniform(0, total_fitness)
            cumulative_fitness = 0
            
            for i, individual in enumerate(population.individuals):
                cumulative_fitness += scaled_fitness[i]
                if cumulative_fitness >= spin:
                    selected.append(individual)
                    break
        
        return selected
    
    def _scale_fitness(self, fitness_values: List[float]) -> List[float]:
        """
        Scale fitness values to be non-negative.
        
        Args:
            fitness_values: Original fitness values
            
        Returns:
            Scaled fitness values (all non-negative)
        """
        if self.scaling_method == 'linear':
            # Linear scaling: shift all values to be positive
            min_fitness = min(fitness_values)
            if min_fitness < 0:
                return [f - min_fitness + 1 for f in fitness_values]
            else:
                return fitness_values
        
        elif self.scaling_method == 'exponential':
            # Exponential scaling: e^(fitness / temperature)
            temperature = max(fitness_values) / 10 if fitness_values else 1
            return [math.exp(f / temperature) for f in fitness_values]
        
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def __str__(self) -> str:
        return f"RouletteWheelSelection(scaling={self.scaling_method})"


class RankSelection(SelectionOperator):
    """
    Rank Selection: Selection based on fitness rank, not raw fitness values.
    
    Instead of using raw fitness values, rank selection:
    1. Sorts individuals by fitness
    2. Assigns ranks (1 = worst, N = best)
    3. Selects based on rank probabilities
    
    This addresses problems with roulette wheel selection:
    - Eliminates sensitivity to fitness scaling
    - Prevents domination by super-fit individuals
    - Works well with negative fitness values
    - Maintains consistent selection pressure
    
    Selection pressure can be controlled by the selection pressure parameter.
    """
    
    def __init__(self, selection_pressure: float = 2.0):
        """
        Initialize rank selection.
        
        Args:
            selection_pressure: Controls selection pressure (1.0 to 2.0)
                              1.0 = uniform selection (no pressure)
                              2.0 = maximum pressure (linear ranking)
        """
        if not 1.0 <= selection_pressure <= 2.0:
            raise ValueError("Selection pressure must be between 1.0 and 2.0")
        self.selection_pressure = selection_pressure
    
    def select(self, population: Population, count: int) -> List[Individual]:
        """
        Perform rank selection.
        
        Args:
            population: Population to select from
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        # Sort individuals by fitness (worst to best)
        sorted_individuals = sorted(population.individuals,
                                  key=lambda ind: ind.get_fitness())
        
        # Calculate rank-based probabilities
        n = len(sorted_individuals)
        probabilities = []
        
        for rank in range(1, n + 1):  # rank goes from 1 (worst) to n (best)
            # Linear ranking formula
            prob = (2 - self.selection_pressure + 
                   2 * (self.selection_pressure - 1) * (rank - 1) / (n - 1)) / n
            probabilities.append(prob)
        
        # Select individuals based on probabilities
        selected = []
        for _ in range(count):
            # Weighted random selection
            spin = random.random()
            cumulative_prob = 0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if cumulative_prob >= spin:
                    selected.append(sorted_individuals[i])
                    break
        
        return selected
    
    def __str__(self) -> str:
        return f"RankSelection(pressure={self.selection_pressure})"


class StochasticUniversalSampling(SelectionOperator):
    """
    Stochastic Universal Sampling (SUS): Improved version of roulette wheel.
    
    SUS addresses the high variance problem of roulette wheel selection by:
    - Using evenly spaced pointers instead of random spins
    - Guaranteeing that selection matches expected values more closely
    - Reducing selection noise while maintaining proportionate selection
    
    Instead of multiple random spins, SUS:
    1. Divides the wheel into N equal segments
    2. Uses one random starting point
    3. Places pointers at fixed intervals
    
    This results in lower variance and more predictable selection behavior.
    """
    
    def __init__(self, scaling_method: str = 'linear'):
        """
        Initialize stochastic universal sampling.
        
        Args:
            scaling_method: Method to handle negative fitness values
        """
        self.scaling_method = scaling_method
    
    def select(self, population: Population, count: int) -> List[Individual]:
        """
        Perform stochastic universal sampling.
        
        Args:
            population: Population to select from
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        # Get fitness values and handle negative values
        fitness_values = [ind.get_fitness() for ind in population.individuals]
        scaled_fitness = self._scale_fitness(fitness_values)
        
        total_fitness = sum(scaled_fitness)
        
        if total_fitness == 0:
            return random.choices(population.individuals, k=count)
        
        # Calculate pointer spacing
        pointer_distance = total_fitness / count
        start = random.uniform(0, pointer_distance)
        
        # Generate evenly spaced pointers
        pointers = [start + i * pointer_distance for i in range(count)]
        
        selected = []
        cumulative_fitness = 0
        individual_index = 0
        
        for pointer in pointers:
            # Advance to the individual that contains this pointer
            while cumulative_fitness < pointer and individual_index < len(population.individuals):
                cumulative_fitness += scaled_fitness[individual_index]
                individual_index += 1
            
            # Select the current individual (or last one if we've gone past the end)
            selected_index = min(individual_index - 1, len(population.individuals) - 1)
            selected.append(population.individuals[selected_index])
        
        return selected
    
    def _scale_fitness(self, fitness_values: List[float]) -> List[float]:
        """Scale fitness values to be non-negative."""
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            return [f - min_fitness + 1 for f in fitness_values]
        else:
            return fitness_values
    
    def __str__(self) -> str:
        return f"StochasticUniversalSampling(scaling={self.scaling_method})"


class TruncationSelection(SelectionOperator):
    """
    Truncation Selection: Select only the best individuals.
    
    This is the simplest and most aggressive selection method:
    1. Sort population by fitness
    2. Select only the top T% of individuals
    3. Fill remaining slots by repeating the selected individuals
    
    Characteristics:
    - Very high selection pressure
    - Fast convergence
    - High risk of premature convergence
    - Loss of diversity
    
    Best used when:
    - You want very fast convergence
    - Population diversity is maintained by other means
    - In combination with high mutation rates
    """
    
    def __init__(self, truncation_threshold: float = 0.5):
        """
        Initialize truncation selection.
        
        Args:
            truncation_threshold: Fraction of population to select (0.0 to 1.0)
                                Common values: 0.5 (select top 50%), 0.3, 0.7
        """
        if not 0.0 < truncation_threshold <= 1.0:
            raise ValueError("Truncation threshold must be between 0.0 and 1.0")
        self.truncation_threshold = truncation_threshold
    
    def select(self, population: Population, count: int) -> List[Individual]:
        """
        Perform truncation selection.
        
        Args:
            population: Population to select from
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        # Sort population by fitness (best first)
        sorted_individuals = sorted(population.individuals,
                                  key=lambda ind: ind.get_fitness(),
                                  reverse=True)
        
        # Calculate how many individuals to keep
        keep_count = max(1, int(len(sorted_individuals) * self.truncation_threshold))
        selected_pool = sorted_individuals[:keep_count]
        
        # Fill the selection by repeating individuals from the pool
        selected = []
        for i in range(count):
            selected.append(selected_pool[i % len(selected_pool)])
        
        return selected
    
    def __str__(self) -> str:
        return f"TruncationSelection(threshold={self.truncation_threshold})"


# Convenience functions for common selection configurations

def tournament_selection(tournament_size: int = 3) -> TournamentSelection:
    """Create a tournament selection operator."""
    return TournamentSelection(tournament_size)


def roulette_wheel_selection(scaling_method: str = 'linear') -> RouletteWheelSelection:
    """Create a roulette wheel selection operator."""
    return RouletteWheelSelection(scaling_method)


def rank_selection(selection_pressure: float = 2.0) -> RankSelection:
    """Create a rank selection operator."""
    return RankSelection(selection_pressure)


def sus_selection(scaling_method: str = 'linear') -> StochasticUniversalSampling:
    """Create a stochastic universal sampling operator."""
    return StochasticUniversalSampling(scaling_method)


def truncation_selection(threshold: float = 0.5) -> TruncationSelection:
    """Create a truncation selection operator."""
    return TruncationSelection(threshold)