from typing import List, Type, Union
import random
import math
from .individual import Individual, BinaryIndividual, PermutationIndividual, RealValuedIndividual


class MutationOperator:
    """
    Base class for mutation operators in genetic algorithms.
    
    Mutation introduces random changes to individual genes, serving several purposes:
    - Maintains genetic diversity in the population
    - Prevents premature convergence to local optima  
    - Enables exploration of new regions in the search space
    - Acts as a "background operator" to crossover
    
    Key concepts:
    - Mutation rate: Probability of mutating each gene (typically 0.001 to 0.1)
    - Mutation strength: How much a gene changes when mutated
    - Adaptive mutation: Changing mutation parameters during evolution
    
    The balance between mutation and selection is crucial:
    - Too little mutation: Population converges prematurely
    - Too much mutation: Algorithm becomes random search
    """
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply mutation to an individual.
        
        Args:
            individual: Individual to mutate (modified in place)
            mutation_rate: Probability of mutating each gene
        """
        raise NotImplementedError("Subclasses must implement mutate method")


class BitFlipMutation(MutationOperator):
    """
    Bit Flip Mutation: For binary representations.
    
    This is the standard mutation operator for binary strings:
    - For each bit, flip it with probability = mutation_rate
    - 0 becomes 1, 1 becomes 0
    
    Example with mutation_rate = 0.1:
    Original: [1, 0, 1, 1, 0, 0, 1, 0]
    Mutated:  [1, 1, 1, 1, 0, 0, 1, 0]  (bit 1 flipped)
    
    Characteristics:
    - Simple and effective for binary problems
    - Each gene has independent mutation probability
    - Maintains diversity in binary populations
    - Works well with crossover operators
    """
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply bit flip mutation to a binary individual.
        
        Args:
            individual: Binary individual to mutate
            mutation_rate: Probability of flipping each bit
        """
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                # Flip the bit
                individual.genes[i] = 1 - individual.genes[i]
        
        # Fitness is no longer valid after mutation
        individual.invalidate_fitness()
    
    def __str__(self) -> str:
        return "BitFlipMutation"


class SwapMutation(MutationOperator):
    """
    Swap Mutation: For permutation representations.
    
    Randomly selects two positions and swaps their values.
    This maintains the permutation property (no duplicates, all elements present).
    
    Example:
    Original: [3, 1, 4, 2, 5, 0]
    Swap positions 1 and 4: [3, 5, 4, 2, 1, 0]
    
    Characteristics:
    - Maintains valid permutations
    - Small change (only 2 positions affected)
    - Good for problems where order matters (TSP, scheduling)
    - Can be applied multiple times for larger changes
    """
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply swap mutation to a permutation individual.
        
        Args:
            individual: Permutation individual to mutate
            mutation_rate: Probability of performing a swap
        """
        if random.random() < mutation_rate and len(individual.genes) >= 2:
            # Select two random positions to swap
            pos1, pos2 = random.sample(range(len(individual.genes)), 2)
            
            # Swap the elements
            individual.genes[pos1], individual.genes[pos2] = \
                individual.genes[pos2], individual.genes[pos1]
            
            individual.invalidate_fitness()
    
    def __str__(self) -> str:
        return "SwapMutation"


class InsertionMutation(MutationOperator):
    """
    Insertion Mutation: For permutation representations.
    
    Removes an element from one position and inserts it at another position.
    This creates a larger change than swap mutation while maintaining permutation validity.
    
    Example:
    Original: [3, 1, 4, 2, 5, 0]
    Remove element at position 2 (value 4) and insert at position 0:
    Result:   [4, 3, 1, 2, 5, 0]
    
    Characteristics:
    - Maintains valid permutations
    - Larger change than swap mutation
    - Good for problems where subsequence order is important
    - Can help escape local optima in routing problems
    """
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply insertion mutation to a permutation individual.
        
        Args:
            individual: Permutation individual to mutate
            mutation_rate: Probability of performing an insertion
        """
        if random.random() < mutation_rate and len(individual.genes) >= 2:
            # Select positions for removal and insertion
            remove_pos = random.randint(0, len(individual.genes) - 1)
            insert_pos = random.randint(0, len(individual.genes) - 1)
            
            if remove_pos != insert_pos:
                # Remove element from remove_pos
                element = individual.genes.pop(remove_pos)
                
                # Adjust insert_pos if necessary
                if insert_pos > remove_pos:
                    insert_pos -= 1
                
                # Insert element at insert_pos
                individual.genes.insert(insert_pos, element)
                
                individual.invalidate_fitness()
    
    def __str__(self) -> str:
        return "InsertionMutation"


class InversionMutation(MutationOperator):
    """
    Inversion Mutation: For permutation representations.
    
    Selects a random segment and reverses the order of elements within it.
    This is particularly useful for problems where the relative order of elements matters.
    
    Example:
    Original: [3, 1, 4, 2, 5, 0]
    Invert segment from position 1 to 4: [3, 5, 2, 4, 1, 0]
    
    Characteristics:
    - Maintains valid permutations
    - Can make significant changes with one operation
    - Particularly good for TSP and similar routing problems
    - Preserves adjacency relationships at segment boundaries
    """
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply inversion mutation to a permutation individual.
        
        Args:
            individual: Permutation individual to mutate
            mutation_rate: Probability of performing an inversion
        """
        if random.random() < mutation_rate and len(individual.genes) >= 2:
            # Select random segment to invert
            start = random.randint(0, len(individual.genes) - 2)
            end = random.randint(start + 1, len(individual.genes) - 1)
            
            # Reverse the segment
            individual.genes[start:end + 1] = individual.genes[start:end + 1][::-1]
            
            individual.invalidate_fitness()
    
    def __str__(self) -> str:
        return "InversionMutation"


class GaussianMutation(MutationOperator):
    """
    Gaussian Mutation: For real-valued representations.
    
    Adds Gaussian (normal) noise to each gene with given probability.
    The noise has mean=0 and configurable standard deviation.
    
    New_value = Old_value + N(0, σ)
    
    Where N(0, σ) is a normal distribution with mean 0 and standard deviation σ.
    
    Example:
    Original: [1.5, -0.3, 2.7]
    With σ=0.1: [1.47, -0.35, 2.73]  (small changes)
    With σ=1.0: [2.1, 0.4, 1.9]     (larger changes)
    
    Characteristics:
    - Natural for continuous optimization
    - Small changes more likely than large changes
    - Standard deviation controls mutation strength
    - Can be adaptive (σ changes during evolution)
    """
    
    def __init__(self, sigma: float = 0.1):
        """
        Initialize Gaussian mutation.
        
        Args:
            sigma: Standard deviation of Gaussian noise
                  Typical values: 0.01 to 1.0 depending on problem scale
        """
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        self.sigma = sigma
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply Gaussian mutation to a real-valued individual.
        
        Args:
            individual: Real-valued individual to mutate
            mutation_rate: Probability of mutating each gene
        """
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                # Add Gaussian noise
                noise = random.gauss(0, self.sigma)
                individual.genes[i] += noise
                
                # Apply bounds if individual has them
                if hasattr(individual, 'bounds') and individual.bounds:
                    min_val, max_val = individual.bounds[i]
                    individual.genes[i] = max(min_val, min(max_val, individual.genes[i]))
        
        individual.invalidate_fitness()
    
    def __str__(self) -> str:
        return f"GaussianMutation(sigma={self.sigma})"


class UniformMutation(MutationOperator):
    """
    Uniform Mutation: For real-valued representations.
    
    Replaces each gene with a random value from a uniform distribution
    within the gene's valid range.
    
    This creates larger, more disruptive changes compared to Gaussian mutation.
    
    Example with bounds [-10, 10]:
    Original: [1.5, -0.3, 2.7]
    Mutated:  [7.2, -0.3, -4.1]  (two genes randomly replaced)
    
    Characteristics:
    - More disruptive than Gaussian mutation
    - Uniform distribution over valid range
    - Good for escaping local optima
    - May reduce convergence speed
    """
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply uniform mutation to a real-valued individual.
        
        Args:
            individual: Real-valued individual to mutate
            mutation_rate: Probability of mutating each gene
        """
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                # Get bounds for this gene
                if hasattr(individual, 'bounds') and individual.bounds:
                    min_val, max_val = individual.bounds[i]
                    individual.genes[i] = random.uniform(min_val, max_val)
                else:
                    # Default bounds if not specified
                    individual.genes[i] = random.uniform(-10.0, 10.0)
        
        individual.invalidate_fitness()
    
    def __str__(self) -> str:
        return "UniformMutation"


class PolynomialMutation(MutationOperator):
    """
    Polynomial Mutation: For real-valued representations.
    
    Uses a polynomial probability distribution to generate mutations.
    This gives higher probability to small changes and lower probability to large changes,
    similar to Gaussian mutation but with different distribution shape.
    
    The distribution index parameter controls the shape:
    - High values: More small changes (similar to low σ Gaussian)
    - Low values: More large changes (more exploration)
    
    Characteristics:
    - Self-adaptive (no need to set mutation strength)
    - Bounded (respects gene boundaries automatically)
    - Good balance between exploitation and exploration
    - Popular in modern evolutionary algorithms
    """
    
    def __init__(self, distribution_index: float = 20.0):
        """
        Initialize polynomial mutation.
        
        Args:
            distribution_index: Controls mutation distribution shape
                              Common values: 5-100 (20 is typical)
        """
        if distribution_index <= 0:
            raise ValueError("Distribution index must be positive")
        self.distribution_index = distribution_index
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply polynomial mutation to a real-valued individual.
        
        Args:
            individual: Real-valued individual to mutate
            mutation_rate: Probability of mutating each gene
        """
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                # Get bounds for this gene
                if hasattr(individual, 'bounds') and individual.bounds:
                    min_val, max_val = individual.bounds[i]
                else:
                    min_val, max_val = -10.0, 10.0
                
                gene_value = individual.genes[i]
                
                # Calculate polynomial mutation
                delta1 = (gene_value - min_val) / (max_val - min_val)
                delta2 = (max_val - gene_value) / (max_val - min_val)
                
                rnd = random.random()
                mut_pow = 1.0 / (self.distribution_index + 1.0)
                
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.distribution_index + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.distribution_index + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                # Apply mutation
                mutated_value = gene_value + deltaq * (max_val - min_val)
                
                # Ensure bounds
                individual.genes[i] = max(min_val, min(max_val, mutated_value))
        
        individual.invalidate_fitness()
    
    def __str__(self) -> str:
        return f"PolynomialMutation(eta={self.distribution_index})"


class AdaptiveMutation(MutationOperator):
    """
    Adaptive Mutation: Adjusts mutation parameters during evolution.
    
    The idea is to use high mutation rates early in evolution (for exploration)
    and lower rates later (for fine-tuning). This can be implemented in various ways:
    
    1. Linear decay: mutation_rate = initial_rate * (1 - generation/max_generations)
    2. Exponential decay: mutation_rate = initial_rate * exp(-decay * generation)
    3. Success-based: Increase/decrease based on improvement rate
    
    This implementation uses exponential decay as it's simple and effective.
    """
    
    def __init__(self, base_mutation: MutationOperator, 
                 initial_rate: float = 0.1, 
                 decay_factor: float = 0.95):
        """
        Initialize adaptive mutation.
        
        Args:
            base_mutation: Underlying mutation operator to adapt
            initial_rate: Starting mutation rate
            decay_factor: Rate decay per generation (0.9-0.99 typical)
        """
        self.base_mutation = base_mutation
        self.initial_rate = initial_rate
        self.decay_factor = decay_factor
        self.current_rate = initial_rate
        self.generation = 0
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply adaptive mutation.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Base mutation rate (will be adapted)
        """
        # Use adaptive rate instead of provided rate
        self.base_mutation.mutate(individual, self.current_rate)
    
    def update_generation(self):
        """Update the mutation rate for the next generation."""
        self.generation += 1
        self.current_rate = self.initial_rate * (self.decay_factor ** self.generation)
    
    def __str__(self) -> str:
        return f"AdaptiveMutation({self.base_mutation}, rate={self.current_rate:.4f})"


# Convenience functions for creating mutation operators

def bit_flip_mutation() -> BitFlipMutation:
    """Create a bit flip mutation operator."""
    return BitFlipMutation()


def swap_mutation() -> SwapMutation:
    """Create a swap mutation operator."""
    return SwapMutation()


def insertion_mutation() -> InsertionMutation:
    """Create an insertion mutation operator."""
    return InsertionMutation()


def inversion_mutation() -> InversionMutation:
    """Create an inversion mutation operator."""
    return InversionMutation()


def gaussian_mutation(sigma: float = 0.1) -> GaussianMutation:
    """Create a Gaussian mutation operator."""
    return GaussianMutation(sigma)


def uniform_mutation() -> UniformMutation:
    """Create a uniform mutation operator."""
    return UniformMutation()


def polynomial_mutation(distribution_index: float = 20.0) -> PolynomialMutation:
    """Create a polynomial mutation operator."""
    return PolynomialMutation(distribution_index)


# Automatic mutation operator selection based on individual type

def get_default_mutation(individual_type: Type[Individual]) -> MutationOperator:
    """
    Get a sensible default mutation operator for a given individual type.
    
    Args:
        individual_type: The type of individual
        
    Returns:
        Appropriate mutation operator
    """
    if issubclass(individual_type, PermutationIndividual):
        return SwapMutation()
    elif issubclass(individual_type, RealValuedIndividual):
        return GaussianMutation()
    elif issubclass(individual_type, BinaryIndividual):
        return BitFlipMutation()
    else:
        # Default to bit flip for unknown types
        return BitFlipMutation()


class CompositeMutation(MutationOperator):
    """
    Composite Mutation: Combines multiple mutation operators.
    
    Sometimes it's beneficial to use multiple mutation operators:
    - Apply different operators with different probabilities
    - Use different operators for different genes
    - Combine complementary mutation strategies
    
    This operator randomly selects one of the provided operators
    for each mutation event.
    """
    
    def __init__(self, operators: List[MutationOperator], weights: List[float] = None):
        """
        Initialize composite mutation.
        
        Args:
            operators: List of mutation operators to choose from
            weights: Probability weights for each operator (default: equal)
        """
        if not operators:
            raise ValueError("At least one mutation operator required")
        
        self.operators = operators
        self.weights = weights or [1.0] * len(operators)
        
        if len(self.weights) != len(operators):
            raise ValueError("Number of weights must match number of operators")
    
    def mutate(self, individual: Individual, mutation_rate: float):
        """
        Apply composite mutation by randomly selecting an operator.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Mutation rate to pass to selected operator
        """
        # Select an operator based on weights
        selected_operator = random.choices(self.operators, weights=self.weights)[0]
        
        # Apply the selected operator
        selected_operator.mutate(individual, mutation_rate)
    
    def __str__(self) -> str:
        operator_names = [str(op) for op in self.operators]
        return f"CompositeMutation({', '.join(operator_names)})"