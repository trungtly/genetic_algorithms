"""
Genetic Algorithms Core Module

This module contains the fundamental components of genetic algorithms:
- Individual: Abstract base class for solution representations
- Population: Management of collections of individuals
- Selection: Methods for choosing parents for reproduction
- Crossover: Operators for combining genetic material
- Mutation: Operators for introducing variation
- GeneticAlgorithm: Main algorithm orchestrator

The module is designed to be educational, with extensive documentation
and clear separation of concerns for easy understanding and extension.
"""

from .individual import (
    Individual,
    BinaryIndividual,
    PermutationIndividual,
    RealValuedIndividual
)

from .population import Population

from .selection import (
    SelectionOperator,
    TournamentSelection,
    RouletteWheelSelection,
    RankSelection,
    StochasticUniversalSampling,
    TruncationSelection,
    tournament_selection,
    roulette_wheel_selection,
    rank_selection,
    sus_selection,
    truncation_selection
)

from .crossover import (
    CrossoverOperator,
    SinglePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
    OrderCrossover,
    PartiallyMappedCrossover,
    ArithmeticCrossover,
    BlendCrossover,
    single_point_crossover,
    two_point_crossover,
    uniform_crossover,
    order_crossover,
    pmx_crossover,
    arithmetic_crossover,
    blend_crossover,
    get_default_crossover
)

from .mutation import (
    MutationOperator,
    BitFlipMutation,
    SwapMutation,
    InsertionMutation,
    InversionMutation,
    GaussianMutation,
    UniformMutation,
    PolynomialMutation,
    AdaptiveMutation,
    CompositeMutation,
    bit_flip_mutation,
    swap_mutation,
    insertion_mutation,
    inversion_mutation,
    gaussian_mutation,
    uniform_mutation,
    polynomial_mutation,
    get_default_mutation
)

from .genetic_algorithm import (
    GeneticAlgorithm,
    solve_with_ga
)

__all__ = [
    # Individual classes
    'Individual',
    'BinaryIndividual', 
    'PermutationIndividual',
    'RealValuedIndividual',
    
    # Population management
    'Population',
    
    # Selection operators
    'SelectionOperator',
    'TournamentSelection',
    'RouletteWheelSelection', 
    'RankSelection',
    'StochasticUniversalSampling',
    'TruncationSelection',
    'tournament_selection',
    'roulette_wheel_selection',
    'rank_selection',
    'sus_selection', 
    'truncation_selection',
    
    # Crossover operators
    'CrossoverOperator',
    'SinglePointCrossover',
    'TwoPointCrossover',
    'UniformCrossover',
    'OrderCrossover',
    'PartiallyMappedCrossover',
    'ArithmeticCrossover',
    'BlendCrossover',
    'single_point_crossover',
    'two_point_crossover', 
    'uniform_crossover',
    'order_crossover',
    'pmx_crossover',
    'arithmetic_crossover',
    'blend_crossover',
    'get_default_crossover',
    
    # Mutation operators
    'MutationOperator',
    'BitFlipMutation',
    'SwapMutation',
    'InsertionMutation', 
    'InversionMutation',
    'GaussianMutation',
    'UniformMutation',
    'PolynomialMutation',
    'AdaptiveMutation',
    'CompositeMutation',
    'bit_flip_mutation',
    'swap_mutation',
    'insertion_mutation',
    'inversion_mutation',
    'gaussian_mutation',
    'uniform_mutation',
    'polynomial_mutation',
    'get_default_mutation',
    
    # Main algorithm
    'GeneticAlgorithm',
    'solve_with_ga'
]