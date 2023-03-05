from typing import List, Tuple, Optional, Type
import random
import copy
from .individual import Individual, BinaryIndividual, PermutationIndividual, RealValuedIndividual


class CrossoverOperator:
    """
    Base class for crossover operators in genetic algorithms.
    
    Crossover (also called recombination) is the process of combining
    genetic material from two or more parent individuals to create
    offspring. This is the primary mechanism for exploring new solutions
    by combining good features from different parents.
    
    Key concepts:
    - Parents: Individuals selected for reproduction
    - Offspring: New individuals created by combining parent genes
    - Crossover rate: Probability that crossover occurs (vs. copying parents)
    - Crossover points: Locations where genetic material is exchanged
    
    Different representations require different crossover operators:
    - Binary strings: Single-point, two-point, uniform crossover
    - Permutations: Order crossover, partially mapped crossover
    - Real values: Arithmetic crossover, blend crossover
    """
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents to create two offspring.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        raise NotImplementedError("Subclasses must implement crossover method")


class SinglePointCrossover(CrossoverOperator):
    """
    Single-Point Crossover: Split parents at one point and exchange tails.
    
    This is the simplest and most common crossover operator:
    1. Choose a random crossover point
    2. Copy genes before the point from parent1 to offspring1
    3. Copy genes after the point from parent2 to offspring1
    4. Do the reverse for offspring2
    
    Example with binary strings:
    Parent1: [1,1,0,0,1,1,0]  point=3
    Parent2: [0,0,1,1,0,0,1]
    
    Offspring1: [1,1,0|1,0,0,1]
    Offspring2: [0,0,1|0,1,1,0]
    
    Characteristics:
    - Simple and fast
    - Works well for many problems
    - May break up good building blocks
    - Building block disruption depends on crossover point location
    """
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring created by single-point crossover
        """
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parents must have the same gene length")
        
        if len(parent1.genes) <= 1:
            # Can't perform crossover on single gene, return copies
            return parent1.copy(), parent2.copy()
        
        # Choose crossover point (1 to length-1)
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        
        # Create offspring genes
        offspring1_genes = (parent1.genes[:crossover_point] + 
                           parent2.genes[crossover_point:])
        offspring2_genes = (parent2.genes[:crossover_point] + 
                           parent1.genes[crossover_point:])
        
        # Create offspring individuals
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        
        offspring1.genes = offspring1_genes
        offspring2.genes = offspring2_genes
        
        # Invalidate fitness (genes have changed)
        offspring1.invalidate_fitness()
        offspring2.invalidate_fitness()
        
        return offspring1, offspring2
    
    def __str__(self) -> str:
        return "SinglePointCrossover"


class TwoPointCrossover(CrossoverOperator):
    """
    Two-Point Crossover: Exchange the middle segment between two points.
    
    This crossover operator:
    1. Choose two random crossover points
    2. Exchange the middle segment between parents
    3. Keep the head and tail segments from the original parent
    
    Example:
    Parent1: [1,1,0,0,1,1,0]  points=2,5
    Parent2: [0,0,1,1,0,0,1]
    
    Offspring1: [1,1|1,1,0|1,0]
    Offspring2: [0,0|0,0,1|0,1]
    
    Advantages over single-point:
    - Better preserves building blocks at the ends
    - More flexible gene exchange
    - Often better performance on many problems
    """
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform two-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring created by two-point crossover
        """
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parents must have the same gene length")
        
        if len(parent1.genes) <= 2:
            # Not enough genes for two-point crossover, use single-point
            return SinglePointCrossover().crossover(parent1, parent2)
        
        # Choose two crossover points and sort them
        point1 = random.randint(1, len(parent1.genes) - 2)
        point2 = random.randint(point1 + 1, len(parent1.genes) - 1)
        
        # Create offspring genes
        offspring1_genes = (parent1.genes[:point1] + 
                           parent2.genes[point1:point2] + 
                           parent1.genes[point2:])
        offspring2_genes = (parent2.genes[:point1] + 
                           parent1.genes[point1:point2] + 
                           parent2.genes[point2:])
        
        # Create offspring individuals
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        
        offspring1.genes = offspring1_genes
        offspring2.genes = offspring2_genes
        
        offspring1.invalidate_fitness()
        offspring2.invalidate_fitness()
        
        return offspring1, offspring2
    
    def __str__(self) -> str:
        return "TwoPointCrossover"


class UniformCrossover(CrossoverOperator):
    """
    Uniform Crossover: Randomly choose genes from either parent.
    
    For each gene position:
    1. Flip a coin (or use probability threshold)
    2. If heads: take gene from parent1
    3. If tails: take gene from parent2
    
    Example (with 50% probability):
    Parent1: [1,1,0,0,1,1,0]
    Parent2: [0,0,1,1,0,0,1]
    Mask:    [1,0,1,0,0,1,1]  (1=parent1, 0=parent2)
    
    Offspring1: [1,0,0,1,0,1,0]
    Offspring2: [0,1,1,0,1,0,1]
    
    Characteristics:
    - High disruption of building blocks
    - Good mixing of genetic material
    - Can be too disruptive for some problems
    - Parameter p controls exchange probability
    """
    
    def __init__(self, exchange_probability: float = 0.5):
        """
        Initialize uniform crossover.
        
        Args:
            exchange_probability: Probability of taking gene from first parent
                                Common values: 0.5 (equal probability)
        """
        if not 0.0 <= exchange_probability <= 1.0:
            raise ValueError("Exchange probability must be between 0.0 and 1.0")
        self.exchange_probability = exchange_probability
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring created by uniform crossover
        """
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parents must have the same gene length")
        
        offspring1_genes = []
        offspring2_genes = []
        
        for i in range(len(parent1.genes)):
            if random.random() < self.exchange_probability:
                # Take from parent1 for offspring1, parent2 for offspring2
                offspring1_genes.append(parent1.genes[i])
                offspring2_genes.append(parent2.genes[i])
            else:
                # Take from parent2 for offspring1, parent1 for offspring2
                offspring1_genes.append(parent2.genes[i])
                offspring2_genes.append(parent1.genes[i])
        
        # Create offspring individuals
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        
        offspring1.genes = offspring1_genes
        offspring2.genes = offspring2_genes
        
        offspring1.invalidate_fitness()
        offspring2.invalidate_fitness()
        
        return offspring1, offspring2
    
    def __str__(self) -> str:
        return f"UniformCrossover(p={self.exchange_probability})"


class OrderCrossover(CrossoverOperator):
    """
    Order Crossover (OX): For permutation-based representations.
    
    Used for problems where genes represent an ordering (e.g., TSP).
    The challenge is maintaining valid permutations (no duplicates, no missing elements).
    
    Algorithm:
    1. Select a random segment from parent1
    2. Copy this segment to offspring1 at the same positions
    3. Fill remaining positions with elements from parent2 in their order of appearance
    4. Skip elements already present in the segment
    
    Example (TSP cities):
    Parent1: [1,2,3,4,5,6,7,8]  segment: positions 3-5
    Parent2: [2,4,6,8,7,5,3,1]
    
    Offspring1: [_,_,_,4,5,6,_,_]  (copy segment)
    Parent2 order: 2,4,6,8,7,5,3,1  (skip 4,5,6 already used)
    Final: [2,8,7,4,5,6,3,1]
    
    This preserves relative order from parent2 while keeping a segment from parent1.
    """
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform order crossover.
        
        Args:
            parent1: First parent (must be PermutationIndividual)
            parent2: Second parent (must be PermutationIndividual)
            
        Returns:
            Two offspring created by order crossover
        """
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parents must have the same gene length")
        
        length = len(parent1.genes)
        if length <= 2:
            return parent1.copy(), parent2.copy()
        
        # Choose crossover segment
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        
        # Create offspring
        offspring1 = self._create_ox_offspring(parent1.genes, parent2.genes, start, end)
        offspring2 = self._create_ox_offspring(parent2.genes, parent1.genes, start, end)
        
        # Create individual objects
        result1 = copy.deepcopy(parent1)
        result2 = copy.deepcopy(parent2)
        
        result1.genes = offspring1
        result2.genes = offspring2
        
        result1.invalidate_fitness()
        result2.invalidate_fitness()
        
        return result1, result2
    
    def _create_ox_offspring(self, parent1_genes: List, parent2_genes: List, 
                           start: int, end: int) -> List:
        """
        Create one offspring using order crossover.
        
        Args:
            parent1_genes: Genes from first parent
            parent2_genes: Genes from second parent  
            start: Start of crossover segment
            end: End of crossover segment
            
        Returns:
            Offspring genes
        """
        length = len(parent1_genes)
        offspring = [None] * length
        
        # Copy segment from parent1
        for i in range(start, end + 1):
            offspring[i] = parent1_genes[i]
        
        # Get elements from parent2 not in the segment
        segment_elements = set(parent1_genes[start:end + 1])
        parent2_order = [gene for gene in parent2_genes if gene not in segment_elements]
        
        # Fill remaining positions with parent2 order
        parent2_index = 0
        for i in range(length):
            if offspring[i] is None:
                offspring[i] = parent2_order[parent2_index]
                parent2_index += 1
        
        return offspring
    
    def __str__(self) -> str:
        return "OrderCrossover"


class PartiallyMappedCrossover(CrossoverOperator):
    """
    Partially Mapped Crossover (PMX): Another permutation crossover operator.
    
    PMX creates a mapping between elements in the crossover region
    and uses this mapping to resolve conflicts.
    
    Algorithm:
    1. Choose two crossover points
    2. Copy the middle segment from parent1 to offspring1
    3. Create a mapping between elements in this segment
    4. Use the mapping to place remaining elements while avoiding duplicates
    
    PMX often preserves more adjacency information than OX.
    """
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform partially mapped crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring created by PMX
        """
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parents must have the same gene length")
        
        length = len(parent1.genes)
        if length <= 2:
            return parent1.copy(), parent2.copy()
        
        # Choose crossover points
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        
        # Create offspring
        offspring1 = self._create_pmx_offspring(parent1.genes, parent2.genes, start, end)
        offspring2 = self._create_pmx_offspring(parent2.genes, parent1.genes, start, end)
        
        # Create individual objects
        result1 = copy.deepcopy(parent1)
        result2 = copy.deepcopy(parent2)
        
        result1.genes = offspring1
        result2.genes = offspring2
        
        result1.invalidate_fitness()
        result2.invalidate_fitness()
        
        return result1, result2
    
    def _create_pmx_offspring(self, parent1_genes: List, parent2_genes: List,
                            start: int, end: int) -> List:
        """Create one offspring using PMX."""
        length = len(parent1_genes)
        offspring = parent2_genes[:]  # Start with parent2
        
        # Create mapping between crossover segments
        mapping = {}
        for i in range(start, end + 1):
            if parent1_genes[i] != parent2_genes[i]:
                mapping[parent2_genes[i]] = parent1_genes[i]
        
        # Copy segment from parent1
        for i in range(start, end + 1):
            offspring[i] = parent1_genes[i]
        
        # Resolve conflicts using mapping
        for i in range(length):
            if i < start or i > end:  # Outside crossover region
                element = offspring[i]
                while element in mapping:
                    element = mapping[element]
                offspring[i] = element
        
        return offspring
    
    def __str__(self) -> str:
        return "PartiallyMappedCrossover"


class ArithmeticCrossover(CrossoverOperator):
    """
    Arithmetic Crossover: For real-valued representations.
    
    Creates offspring by taking weighted averages of parent genes:
    offspring1 = α * parent1 + (1-α) * parent2
    offspring2 = (1-α) * parent1 + α * parent2
    
    Where α is the blending parameter (typically 0.5 for equal weighting).
    
    Characteristics:
    - Always produces valid real values
    - Offspring are between parents in the search space
    - Good for continuous optimization problems
    - May reduce diversity if used exclusively
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize arithmetic crossover.
        
        Args:
            alpha: Blending parameter (0.0 to 1.0)
                  0.5 = equal contribution from both parents
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self.alpha = alpha
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform arithmetic crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring created by arithmetic crossover
        """
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parents must have the same gene length")
        
        offspring1_genes = []
        offspring2_genes = []
        
        for i in range(len(parent1.genes)):
            gene1 = parent1.genes[i]
            gene2 = parent2.genes[i]
            
            # Create blended offspring
            offspring1_gene = self.alpha * gene1 + (1 - self.alpha) * gene2
            offspring2_gene = (1 - self.alpha) * gene1 + self.alpha * gene2
            
            offspring1_genes.append(offspring1_gene)
            offspring2_genes.append(offspring2_gene)
        
        # Create offspring individuals
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        
        offspring1.genes = offspring1_genes
        offspring2.genes = offspring2_genes
        
        offspring1.invalidate_fitness()
        offspring2.invalidate_fitness()
        
        return offspring1, offspring2
    
    def __str__(self) -> str:
        return f"ArithmeticCrossover(alpha={self.alpha})"


class BlendCrossover(CrossoverOperator):
    """
    Blend Crossover (BLX-α): Extended arithmetic crossover with exploration.
    
    Instead of staying between parents, BLX-α can produce offspring
    outside the parent range to increase exploration:
    
    For each gene:
    1. Find the range [min_parent, max_parent]
    2. Extend the range by α on both sides
    3. Randomly sample offspring genes from the extended range
    
    This helps maintain diversity and exploration in real-valued optimization.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize blend crossover.
        
        Args:
            alpha: Extension parameter
                  0.0 = no extension (offspring between parents)
                  0.3 = common value, extends range by 30%
        """
        if alpha < 0.0:
            raise ValueError("Alpha must be non-negative")
        self.alpha = alpha
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform blend crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring created by blend crossover
        """
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parents must have the same gene length")
        
        offspring1_genes = []
        offspring2_genes = []
        
        for i in range(len(parent1.genes)):
            gene1 = parent1.genes[i]
            gene2 = parent2.genes[i]
            
            # Calculate extended range
            min_gene = min(gene1, gene2)
            max_gene = max(gene1, gene2)
            range_size = max_gene - min_gene
            
            extended_min = min_gene - self.alpha * range_size
            extended_max = max_gene + self.alpha * range_size
            
            # Sample offspring genes from extended range
            offspring1_gene = random.uniform(extended_min, extended_max)
            offspring2_gene = random.uniform(extended_min, extended_max)
            
            # Apply bounds if parent has them
            if hasattr(parent1, 'bounds') and parent1.bounds:
                bound_min, bound_max = parent1.bounds[i]
                offspring1_gene = max(bound_min, min(bound_max, offspring1_gene))
                offspring2_gene = max(bound_min, min(bound_max, offspring2_gene))
            
            offspring1_genes.append(offspring1_gene)
            offspring2_genes.append(offspring2_gene)
        
        # Create offspring individuals
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        
        offspring1.genes = offspring1_genes
        offspring2.genes = offspring2_genes
        
        offspring1.invalidate_fitness()
        offspring2.invalidate_fitness()
        
        return offspring1, offspring2
    
    def __str__(self) -> str:
        return f"BlendCrossover(alpha={self.alpha})"


# Convenience functions for creating crossover operators

def single_point_crossover() -> SinglePointCrossover:
    """Create a single-point crossover operator."""
    return SinglePointCrossover()


def two_point_crossover() -> TwoPointCrossover:
    """Create a two-point crossover operator."""
    return TwoPointCrossover()


def uniform_crossover(probability: float = 0.5) -> UniformCrossover:
    """Create a uniform crossover operator."""
    return UniformCrossover(probability)


def order_crossover() -> OrderCrossover:
    """Create an order crossover operator for permutations."""
    return OrderCrossover()


def pmx_crossover() -> PartiallyMappedCrossover:
    """Create a PMX crossover operator for permutations."""
    return PartiallyMappedCrossover()


def arithmetic_crossover(alpha: float = 0.5) -> ArithmeticCrossover:
    """Create an arithmetic crossover operator for real values."""
    return ArithmeticCrossover(alpha)


def blend_crossover(alpha: float = 0.3) -> BlendCrossover:
    """Create a blend crossover operator for real values."""
    return BlendCrossover(alpha)


# Automatic crossover operator selection based on individual type

def get_default_crossover(individual_type: Type[Individual]) -> CrossoverOperator:
    """
    Get a sensible default crossover operator for a given individual type.
    
    Args:
        individual_type: The type of individual
        
    Returns:
        Appropriate crossover operator
    """
    if issubclass(individual_type, PermutationIndividual):
        return OrderCrossover()
    elif issubclass(individual_type, RealValuedIndividual):
        return ArithmeticCrossover()
    elif issubclass(individual_type, BinaryIndividual):
        return TwoPointCrossover()
    else:
        # Default to single-point for unknown types
        return SinglePointCrossover()