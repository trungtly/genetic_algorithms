from abc import ABC, abstractmethod
from typing import Any, List, Union
import random
import copy


class Individual(ABC):
    """
    Abstract base class representing an individual in a genetic algorithm population.
    
    An individual represents a potential solution to the optimization problem.
    Each individual has:
    - Genes: The solution representation (chromosome)
    - Fitness: A measure of how good the solution is
    - Age: How many generations this individual has survived
    
    This abstract class defines the interface that all specific problem
    implementations must follow.
    """
    
    def __init__(self, genes: List[Any] = None):
        """
        Initialize an individual.
        
        Args:
            genes: List representing the chromosome/solution.
                  If None, genes should be randomly generated.
        """
        self.genes = genes if genes is not None else self.random_genes()
        self.fitness = None
        self.age = 0
        self._fitness_calculated = False
    
    @abstractmethod
    def random_genes(self) -> List[Any]:
        """
        Generate random genes for this individual.
        
        This method should create a random valid solution representation
        for the specific problem being solved.
        
        Returns:
            List of genes representing a random solution
        """
        pass
    
    @abstractmethod
    def calculate_fitness(self) -> float:
        """
        Calculate and return the fitness of this individual.
        
        Fitness represents how good this solution is. Higher values
        typically indicate better solutions, though this can be
        problem-dependent.
        
        Returns:
            Fitness value as a float
        """
        pass
    
    def get_fitness(self) -> float:
        """
        Get the fitness of this individual, calculating it if necessary.
        
        This method caches the fitness calculation to avoid redundant
        computations, which is important for algorithm efficiency.
        
        Returns:
            Fitness value as a float
        """
        if not self._fitness_calculated:
            self.fitness = self.calculate_fitness()
            self._fitness_calculated = True
        return self.fitness
    
    def invalidate_fitness(self):
        """
        Mark the fitness as invalid, forcing recalculation on next access.
        
        This should be called whenever the genes are modified, as the
        fitness will no longer be accurate.
        """
        self._fitness_calculated = False
        self.fitness = None
    
    def copy(self) -> 'Individual':
        """
        Create a deep copy of this individual.
        
        Returns:
            A new Individual instance with the same genes but independent state
        """
        new_individual = copy.deepcopy(self)
        new_individual.age = 0  # Reset age for the copy
        return new_individual
    
    def mutate(self, mutation_rate: float):
        """
        Apply mutation to this individual's genes.
        
        This is a default implementation that can be overridden for
        problem-specific mutation strategies.
        
        Args:
            mutation_rate: Probability of mutating each gene (0.0 to 1.0)
        """
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = self._mutate_gene(self.genes[i], i)
        
        # Fitness is no longer valid after mutation
        self.invalidate_fitness()
    
    def _mutate_gene(self, gene: Any, index: int) -> Any:
        """
        Mutate a single gene.
        
        This is a default implementation that should be overridden
        for problem-specific gene mutation.
        
        Args:
            gene: The current gene value
            index: The position of the gene in the chromosome
            
        Returns:
            The mutated gene value
        """
        # Default implementation: return a random gene
        # This is not very useful and should be overridden
        return random.choice([True, False]) if isinstance(gene, bool) else gene
    
    def age_increment(self):
        """Increment the age of this individual by one generation."""
        self.age += 1
    
    def __str__(self) -> str:
        """String representation of the individual."""
        fitness_str = f"{self.get_fitness():.4f}" if self.fitness is not None else "N/A"
        return f"Individual(fitness={fitness_str}, age={self.age}, genes={self.genes[:5]}{'...' if len(self.genes) > 5 else ''})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the individual."""
        return f"Individual(genes={self.genes}, fitness={self.fitness}, age={self.age})"
    
    def __lt__(self, other: 'Individual') -> bool:
        """
        Less than comparison based on fitness.
        
        Note: This assumes higher fitness is better. For minimization problems,
        this comparison should be overridden.
        """
        return self.get_fitness() < other.get_fitness()
    
    def __eq__(self, other: 'Individual') -> bool:
        """Equality comparison based on genes."""
        if not isinstance(other, Individual):
            return False
        return self.genes == other.genes


class BinaryIndividual(Individual):
    """
    A concrete implementation of Individual for binary optimization problems.
    
    This class represents solutions as binary strings (lists of 0s and 1s),
    which is useful for many optimization problems like the knapsack problem.
    """
    
    def __init__(self, genes: List[int] = None, length: int = 10):
        """
        Initialize a binary individual.
        
        Args:
            genes: Binary genes (list of 0s and 1s). If None, random genes are generated.
            length: Length of the binary string if genes is None
        """
        self.length = length
        super().__init__(genes)
    
    def random_genes(self) -> List[int]:
        """
        Generate random binary genes.
        
        Returns:
            List of random 0s and 1s of specified length
        """
        return [random.randint(0, 1) for _ in range(self.length)]
    
    def calculate_fitness(self) -> float:
        """
        Default fitness calculation for binary individuals.
        
        This implementation simply counts the number of 1s, which may not
        be appropriate for all problems. Override this method for specific
        problem implementations.
        
        Returns:
            Number of 1s in the binary string
        """
        return sum(self.genes)
    
    def _mutate_gene(self, gene: int, index: int) -> int:
        """
        Mutate a binary gene by flipping it.
        
        Args:
            gene: Current binary value (0 or 1)
            index: Position of the gene (unused in this implementation)
            
        Returns:
            Flipped binary value
        """
        return 1 - gene  # Flip the bit


class PermutationIndividual(Individual):
    """
    A concrete implementation of Individual for permutation-based problems.
    
    This class represents solutions as permutations of elements, which is
    useful for problems like the Traveling Salesman Problem (TSP).
    """
    
    def __init__(self, genes: List[int] = None, size: int = 10):
        """
        Initialize a permutation individual.
        
        Args:
            genes: Permutation genes. If None, random permutation is generated.
            size: Size of the permutation if genes is None
        """
        self.size = size
        super().__init__(genes)
    
    def random_genes(self) -> List[int]:
        """
        Generate a random permutation.
        
        Returns:
            Random permutation of numbers from 0 to size-1
        """
        genes = list(range(self.size))
        random.shuffle(genes)
        return genes
    
    def calculate_fitness(self) -> float:
        """
        Default fitness calculation for permutation individuals.
        
        This is a placeholder implementation that should be overridden
        for specific problems like TSP.
        
        Returns:
            Negative sum of absolute differences between adjacent elements
        """
        fitness = 0
        for i in range(len(self.genes) - 1):
            fitness -= abs(self.genes[i] - self.genes[i + 1])
        return fitness
    
    def _mutate_gene(self, gene: Any, index: int) -> Any:
        """
        For permutations, we don't mutate individual genes.
        Instead, we override the mutate method to use permutation-specific operations.
        """
        return gene
    
    def mutate(self, mutation_rate: float):
        """
        Apply permutation-specific mutation.
        
        Uses swap mutation: randomly swaps two elements in the permutation.
        
        Args:
            mutation_rate: Probability of performing a swap mutation
        """
        if random.random() < mutation_rate:
            # Swap two random positions
            i, j = random.sample(range(len(self.genes)), 2)
            self.genes[i], self.genes[j] = self.genes[j], self.genes[i]
            self.invalidate_fitness()


class RealValuedIndividual(Individual):
    """
    A concrete implementation of Individual for real-valued optimization problems.
    
    This class represents solutions as lists of real numbers, which is useful
    for continuous optimization problems.
    """
    
    def __init__(self, genes: List[float] = None, dimensions: int = 10, 
                 bounds: List[tuple] = None):
        """
        Initialize a real-valued individual.
        
        Args:
            genes: Real-valued genes. If None, random genes are generated.
            dimensions: Number of dimensions if genes is None
            bounds: List of (min, max) tuples for each dimension
        """
        self.dimensions = dimensions
        self.bounds = bounds or [(-10.0, 10.0)] * dimensions
        super().__init__(genes)
    
    def random_genes(self) -> List[float]:
        """
        Generate random real-valued genes within specified bounds.
        
        Returns:
            List of random real numbers within the specified bounds
        """
        genes = []
        for min_val, max_val in self.bounds:
            genes.append(random.uniform(min_val, max_val))
        return genes
    
    def calculate_fitness(self) -> float:
        """
        Default fitness calculation for real-valued individuals.
        
        This implements the sphere function (minimize sum of squares).
        Override this method for specific optimization problems.
        
        Returns:
            Negative sum of squares (for maximization)
        """
        return -sum(x**2 for x in self.genes)
    
    def _mutate_gene(self, gene: float, index: int) -> float:
        """
        Mutate a real-valued gene using Gaussian noise.
        
        Args:
            gene: Current gene value
            index: Position of the gene
            
        Returns:
            Mutated gene value, clipped to bounds
        """
        # Add Gaussian noise
        noise = random.gauss(0, 0.1)  # Mean=0, std=0.1
        new_gene = gene + noise
        
        # Clip to bounds
        min_val, max_val = self.bounds[index]
        return max(min_val, min(max_val, new_gene))