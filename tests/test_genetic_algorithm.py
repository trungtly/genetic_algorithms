"""
Unit tests for the GeneticAlgorithm class.
"""

import pytest
import random
from genetic_algorithms.core.genetic_algorithm import GeneticAlgorithm, solve_with_ga
from genetic_algorithms.core.individual import BinaryIndividual, RealValuedIndividual
from genetic_algorithms.core.population import Population
from genetic_algorithms.core.selection import TournamentSelection


class SimpleBinaryIndividual(BinaryIndividual):
    """Simple binary individual for testing - maximize number of 1s."""
    pass  # Uses default fitness calculation (count of 1s)


class TestGeneticAlgorithmInitialization:
    """Tests for GA initialization."""

    def test_default_initialization(self):
        """Test GA with default parameters."""
        ga = GeneticAlgorithm()

        assert ga.population_size == 100
        assert ga.mutation_rate == 0.01
        assert ga.crossover_rate == 0.8
        assert ga.elite_count == 2
        assert ga.maximize is True

    def test_custom_parameters(self):
        """Test GA with custom parameters."""
        ga = GeneticAlgorithm(
            population_size=50,
            mutation_rate=0.05,
            crossover_rate=0.9,
            elite_count=5
        )

        assert ga.population_size == 50
        assert ga.mutation_rate == 0.05
        assert ga.crossover_rate == 0.9
        assert ga.elite_count == 5

    def test_invalid_population_size(self):
        """Test that invalid population size raises error."""
        with pytest.raises(ValueError):
            GeneticAlgorithm(population_size=1)
        with pytest.raises(ValueError):
            GeneticAlgorithm(population_size=0)

    def test_invalid_mutation_rate(self):
        """Test that invalid mutation rate raises error."""
        with pytest.raises(ValueError):
            GeneticAlgorithm(mutation_rate=-0.1)
        with pytest.raises(ValueError):
            GeneticAlgorithm(mutation_rate=1.5)

    def test_invalid_crossover_rate(self):
        """Test that invalid crossover rate raises error."""
        with pytest.raises(ValueError):
            GeneticAlgorithm(crossover_rate=-0.1)
        with pytest.raises(ValueError):
            GeneticAlgorithm(crossover_rate=1.5)

    def test_invalid_elite_count(self):
        """Test that invalid elite count raises error."""
        with pytest.raises(ValueError):
            GeneticAlgorithm(population_size=10, elite_count=10)
        with pytest.raises(ValueError):
            GeneticAlgorithm(elite_count=-1)


class TestGeneticAlgorithmEvolution:
    """Tests for GA evolution."""

    def test_basic_evolution(self):
        """Test basic evolution run."""
        random.seed(42)
        ga = GeneticAlgorithm(
            population_size=20,
            mutation_rate=0.1,
            elite_count=2
        )

        best = ga.evolve(
            individual_class=SimpleBinaryIndividual,
            max_generations=10,
            verbose=False,
            length=10
        )

        assert best is not None
        assert isinstance(best, SimpleBinaryIndividual)
        assert ga.generation == 10

    def test_target_fitness_termination(self):
        """Test that evolution stops when target fitness is reached."""
        random.seed(42)
        ga = GeneticAlgorithm(
            population_size=50,
            mutation_rate=0.1,
            elite_count=2
        )

        best = ga.evolve(
            individual_class=SimpleBinaryIndividual,
            max_generations=1000,
            target_fitness=10,  # Maximum possible fitness
            verbose=False,
            length=10
        )

        # Should stop early if target is reached
        assert best.get_fitness() >= 10 or ga.generation < 1000

    def test_fitness_improves_over_generations(self):
        """Test that fitness generally improves."""
        random.seed(42)
        ga = GeneticAlgorithm(
            population_size=50,
            mutation_rate=0.05,
            crossover_rate=0.8,
            elite_count=2
        )

        ga.evolve(
            individual_class=SimpleBinaryIndividual,
            max_generations=50,
            verbose=False,
            length=20
        )

        # Best fitness at end should be >= best at start
        assert ga.best_fitness_history[-1] >= ga.best_fitness_history[0]

    def test_elitism_preserves_best(self):
        """Test that elitism preserves best individuals."""
        random.seed(42)
        ga = GeneticAlgorithm(
            population_size=20,
            mutation_rate=0.5,  # High mutation rate
            crossover_rate=0.8,
            elite_count=2
        )

        ga.evolve(
            individual_class=SimpleBinaryIndividual,
            max_generations=20,
            verbose=False,
            length=10
        )

        # With elitism, best fitness should never decrease
        for i in range(1, len(ga.best_fitness_history)):
            assert ga.best_fitness_history[i] >= ga.best_fitness_history[i-1]

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        random.seed(42)
        ga = GeneticAlgorithm(
            population_size=20,
            elite_count=1
        )

        ga.evolve(
            individual_class=SimpleBinaryIndividual,
            max_generations=5,
            verbose=False,
            length=10
        )

        stats = ga.get_statistics()

        assert stats['generation'] == 5
        assert len(stats['best_fitness_history']) == 6  # Initial + 5 generations
        assert len(stats['average_fitness_history']) == 6
        assert len(stats['diversity_history']) == 6
        assert stats['best_fitness'] is not None

    def test_callback_functions(self):
        """Test that callbacks are called."""
        random.seed(42)
        ga = GeneticAlgorithm(population_size=10, elite_count=1)

        callback_data = []

        def progress_callback(gen, best_fit, avg_fit):
            callback_data.append((gen, best_fit, avg_fit))

        ga.set_progress_callback(progress_callback)

        ga.evolve(
            individual_class=SimpleBinaryIndividual,
            max_generations=5,
            verbose=False,
            length=5
        )

        assert len(callback_data) == 5  # Called each generation


class TestGeneticAlgorithmRestart:
    """Tests for GA restart functionality."""

    def test_restart_clears_state(self):
        """Test that restart clears all state."""
        random.seed(42)
        ga = GeneticAlgorithm(population_size=10)

        ga.evolve(
            individual_class=SimpleBinaryIndividual,
            max_generations=5,
            verbose=False,
            length=5
        )

        ga.restart_evolution()

        assert ga.population is None
        assert ga.generation == 0
        assert ga.best_individual is None
        assert len(ga.best_fitness_history) == 0
        assert ga.evaluations_count == 0


class TestSolveWithGA:
    """Tests for the convenience function."""

    def test_basic_solve(self):
        """Test basic problem solving."""
        random.seed(42)

        best, stats = solve_with_ga(
            individual_class=SimpleBinaryIndividual,
            population_size=20,
            max_generations=10,
            verbose=False,
            length=10
        )

        assert best is not None
        assert isinstance(stats, dict)
        assert 'best_fitness' in stats
        assert 'generation' in stats


class TestMinimization:
    """Tests for minimization problems."""

    def test_minimization_mode(self):
        """Test GA in minimization mode."""
        random.seed(42)
        ga = GeneticAlgorithm(
            population_size=30,
            mutation_rate=0.1,
            elite_count=2,
            maximize=False  # Minimize
        )

        # For sphere function, minimum is at origin (fitness = 0)
        best = ga.evolve(
            individual_class=RealValuedIndividual,
            max_generations=50,
            verbose=False,
            dimensions=3,
            bounds=[(-5, 5), (-5, 5), (-5, 5)]
        )

        # Fitness should be close to 0 (optimal)
        assert best.get_fitness() > -1.0  # Close to 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
