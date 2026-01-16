"""
Logging utilities for the genetic algorithms library.

This module provides a configured logger for the library that can be used
throughout the codebase for consistent logging. Users can configure the
logging level and format to suit their needs.
"""

import logging
import sys
from typing import Optional


# Create the library logger
_logger = logging.getLogger('genetic_algorithms')
_logger.setLevel(logging.WARNING)  # Default to WARNING level

# Prevent duplicate handlers if module is imported multiple times
if not _logger.handlers:
    # Create console handler with default formatting
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(logging.DEBUG)

    # Create formatter
    _formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    _console_handler.setFormatter(_formatter)

    # Add handler to logger
    _logger.addHandler(_console_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for use in the genetic algorithms library.

    Args:
        name: Optional name suffix for the logger. If provided, creates
              a child logger named 'genetic_algorithms.{name}'.

    Returns:
        Logger instance configured for the library.

    Example:
        >>> logger = get_logger('selection')
        >>> logger.info('Performing tournament selection')
    """
    if name:
        return logging.getLogger(f'genetic_algorithms.{name}')
    return _logger


def set_log_level(level: int) -> None:
    """
    Set the logging level for the genetic algorithms library.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)

    Example:
        >>> import logging
        >>> set_log_level(logging.DEBUG)  # Enable debug messages
        >>> set_log_level(logging.WARNING)  # Only warnings and errors
    """
    _logger.setLevel(level)


def enable_debug_logging() -> None:
    """
    Enable debug-level logging for detailed algorithm tracing.

    This is useful for debugging and understanding algorithm behavior.
    """
    set_log_level(logging.DEBUG)


def disable_logging() -> None:
    """
    Disable all logging output from the library.

    This is useful when you want silent operation.
    """
    _logger.setLevel(logging.CRITICAL + 1)


def add_file_handler(filepath: str, level: int = logging.DEBUG) -> None:
    """
    Add a file handler to log messages to a file.

    Args:
        filepath: Path to the log file
        level: Logging level for the file handler

    Example:
        >>> add_file_handler('evolution.log', logging.INFO)
    """
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(level)
    file_handler.setFormatter(_formatter)
    _logger.addHandler(file_handler)


class EvolutionLogger:
    """
    A specialized logger for tracking evolution progress.

    This class provides convenient methods for logging common events
    during genetic algorithm execution.

    Example:
        >>> evo_logger = EvolutionLogger()
        >>> evo_logger.log_generation(gen=10, best_fitness=95.5, avg_fitness=75.2)
        >>> evo_logger.log_new_best(fitness=96.0, generation=11)
    """

    def __init__(self, name: str = 'evolution'):
        """
        Initialize the evolution logger.

        Args:
            name: Name suffix for the logger
        """
        self.logger = get_logger(name)

    def log_start(self, population_size: int, max_generations: int) -> None:
        """Log the start of evolution."""
        self.logger.info(
            f"Starting evolution: population_size={population_size}, "
            f"max_generations={max_generations}"
        )

    def log_generation(self, gen: int, best_fitness: float,
                       avg_fitness: float, diversity: float = None) -> None:
        """
        Log statistics for a generation.

        Args:
            gen: Generation number
            best_fitness: Best fitness in the population
            avg_fitness: Average fitness in the population
            diversity: Optional diversity measure
        """
        msg = f"Gen {gen:4d}: best={best_fitness:.6f}, avg={avg_fitness:.6f}"
        if diversity is not None:
            msg += f", diversity={diversity:.6f}"
        self.logger.debug(msg)

    def log_new_best(self, fitness: float, generation: int) -> None:
        """
        Log discovery of a new best individual.

        Args:
            fitness: Fitness of the new best individual
            generation: Generation when found
        """
        self.logger.info(
            f"New best individual found at generation {generation}: "
            f"fitness={fitness:.6f}"
        )

    def log_termination(self, reason: str, generation: int,
                        best_fitness: float, elapsed_time: float) -> None:
        """
        Log evolution termination.

        Args:
            reason: Reason for termination
            generation: Final generation number
            best_fitness: Best fitness achieved
            elapsed_time: Total evolution time in seconds
        """
        self.logger.info(
            f"Evolution terminated: {reason}\n"
            f"  Final generation: {generation}\n"
            f"  Best fitness: {best_fitness:.6f}\n"
            f"  Elapsed time: {elapsed_time:.2f}s"
        )

    def log_convergence_warning(self, generations_without_improvement: int) -> None:
        """
        Log a warning about potential premature convergence.

        Args:
            generations_without_improvement: Number of generations without improvement
        """
        self.logger.warning(
            f"Potential premature convergence: "
            f"{generations_without_improvement} generations without improvement"
        )

    def log_operator_selection(self, selection: str, crossover: str,
                               mutation: str) -> None:
        """
        Log the operators being used.

        Args:
            selection: Selection operator name
            crossover: Crossover operator name
            mutation: Mutation operator name
        """
        self.logger.debug(
            f"Operators: selection={selection}, crossover={crossover}, "
            f"mutation={mutation}"
        )
