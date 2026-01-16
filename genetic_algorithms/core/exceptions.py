"""
Custom exceptions for the genetic algorithms library.

This module defines a hierarchy of exceptions for better error handling
and more informative error messages throughout the library.
"""


class GeneticAlgorithmError(Exception):
    """
    Base exception for all genetic algorithm errors.

    All custom exceptions in this library inherit from this class,
    allowing users to catch all library-specific exceptions with a single
    except clause if desired.
    """
    pass


class ConfigurationError(GeneticAlgorithmError):
    """
    Exception raised for invalid configuration parameters.

    This exception is raised when:
    - Population size is too small
    - Invalid mutation/crossover rates
    - Incompatible operator combinations
    - Invalid termination criteria
    """
    pass


class PopulationError(GeneticAlgorithmError):
    """
    Exception raised for population-related errors.

    This exception is raised when:
    - Population is empty or has insufficient individuals
    - Population replacement fails
    - Selection cannot be performed
    """
    pass


class IndividualError(GeneticAlgorithmError):
    """
    Exception raised for individual-related errors.

    This exception is raised when:
    - Individual creation fails
    - Invalid gene values
    - Fitness calculation errors
    """
    pass


class OperatorError(GeneticAlgorithmError):
    """
    Exception raised for genetic operator errors.

    This exception is raised when:
    - Crossover fails (incompatible parents, etc.)
    - Mutation fails
    - Selection fails
    """
    pass


class CrossoverError(OperatorError):
    """
    Exception raised specifically for crossover operation errors.

    This exception is raised when:
    - Parents have incompatible gene lengths
    - Crossover produces invalid offspring
    - Permutation constraints are violated
    """
    pass


class MutationError(OperatorError):
    """
    Exception raised specifically for mutation operation errors.

    This exception is raised when:
    - Mutation produces invalid genes
    - Bounds are violated
    - Permutation constraints are violated
    """
    pass


class SelectionError(OperatorError):
    """
    Exception raised specifically for selection operation errors.

    This exception is raised when:
    - Population is too small for selection
    - Fitness values are invalid for selection method
    """
    pass


class ConvergenceError(GeneticAlgorithmError):
    """
    Exception raised when evolution fails to progress.

    This exception is raised when:
    - Population converges prematurely
    - No improvement over many generations
    - Fitness becomes stuck
    """
    pass


class TerminationError(GeneticAlgorithmError):
    """
    Exception raised for termination-related errors.

    This exception is raised when:
    - Invalid termination criteria
    - Conflicting termination conditions
    """
    pass


class ProblemDefinitionError(GeneticAlgorithmError):
    """
    Exception raised for problem definition errors.

    This exception is raised when:
    - TSP problem has too few cities
    - Knapsack problem has invalid items or capacity
    - Problem constraints are invalid
    """
    pass


class ValidationError(GeneticAlgorithmError):
    """
    Exception raised when validation fails.

    This exception is raised when:
    - Input parameter validation fails
    - Type checking fails
    - Range checking fails
    """
    pass


def validate_rate(value: float, name: str) -> None:
    """
    Validate that a rate value is between 0.0 and 1.0.

    Args:
        value: The rate value to validate
        name: The name of the parameter (for error message)

    Raises:
        ValidationError: If value is not in [0.0, 1.0]
    """
    if not 0.0 <= value <= 1.0:
        raise ValidationError(
            f"{name} must be between 0.0 and 1.0, got {value}"
        )


def validate_positive(value: float, name: str) -> None:
    """
    Validate that a value is positive.

    Args:
        value: The value to validate
        name: The name of the parameter (for error message)

    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(
            f"{name} must be positive, got {value}"
        )


def validate_non_negative(value: float, name: str) -> None:
    """
    Validate that a value is non-negative.

    Args:
        value: The value to validate
        name: The name of the parameter (for error message)

    Raises:
        ValidationError: If value is negative
    """
    if value < 0:
        raise ValidationError(
            f"{name} must be non-negative, got {value}"
        )


def validate_integer_range(value: int, min_val: int, max_val: int, name: str) -> None:
    """
    Validate that an integer is within a range.

    Args:
        value: The value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: The name of the parameter (for error message)

    Raises:
        ValidationError: If value is not in [min_val, max_val]
    """
    if not min_val <= value <= max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )
