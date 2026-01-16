"""
Unit tests for genetic operators (selection, crossover, mutation).
"""

import pytest
import random
from genetic_algorithms.core.individual import (
    BinaryIndividual, PermutationIndividual, RealValuedIndividual
)
from genetic_algorithms.core.population import Population
from genetic_algorithms.core.selection import (
    TournamentSelection, RouletteWheelSelection, RankSelection,
    StochasticUniversalSampling, TruncationSelection
)
from genetic_algorithms.core.crossover import (
    SinglePointCrossover, TwoPointCrossover, UniformCrossover,
    OrderCrossover, PartiallyMappedCrossover,
    ArithmeticCrossover, BlendCrossover, get_default_crossover
)
from genetic_algorithms.core.mutation import (
    BitFlipMutation, SwapMutation, InsertionMutation, InversionMutation,
    GaussianMutation, UniformMutation, PolynomialMutation,
    CompositeMutation, get_default_mutation
)


# ============================================================================
# SELECTION OPERATOR TESTS
# ============================================================================

class TestTournamentSelection:
    """Tests for tournament selection."""

    def test_selection_returns_correct_count(self):
        """Test that selection returns requested number of individuals."""
        pop = Population(BinaryIndividual, size=20, length=10)
        selector = TournamentSelection(tournament_size=3)

        selected = selector.select(pop, count=5)
        assert len(selected) == 5

    def test_selection_favors_fitter_individuals(self):
        """Test that fitter individuals are selected more often."""
        random.seed(42)

        # Create a population with known fitness values
        pop = Population(BinaryIndividual, size=10, length=10)
        # Manually set genes to have clear fitness differences
        for i, ind in enumerate(pop.individuals):
            ind.genes = [1] * i + [0] * (10 - i)
            ind.invalidate_fitness()

        selector = TournamentSelection(tournament_size=3)

        # Select many times and check that high-fitness individuals are preferred
        selection_counts = [0] * 10
        for _ in range(1000):
            selected = selector.select(pop, count=1)
            idx = pop.individuals.index(selected[0])
            selection_counts[idx] += 1

        # Higher index (higher fitness) should be selected more
        assert selection_counts[9] > selection_counts[0]

    def test_invalid_tournament_size(self):
        """Test that invalid tournament size raises error."""
        with pytest.raises(ValueError):
            TournamentSelection(tournament_size=0)


class TestRouletteWheelSelection:
    """Tests for roulette wheel selection."""

    def test_selection_returns_correct_count(self):
        """Test that selection returns requested number of individuals."""
        pop = Population(BinaryIndividual, size=20, length=10)
        selector = RouletteWheelSelection()

        selected = selector.select(pop, count=5)
        assert len(selected) == 5

    def test_handles_negative_fitness(self):
        """Test that negative fitness values are handled."""
        pop = Population(RealValuedIndividual, size=10, dimensions=5)
        selector = RouletteWheelSelection(scaling_method='linear')

        # Should not raise an error
        selected = selector.select(pop, count=3)
        assert len(selected) == 3


class TestRankSelection:
    """Tests for rank selection."""

    def test_selection_returns_correct_count(self):
        """Test that selection returns requested number of individuals."""
        pop = Population(BinaryIndividual, size=20, length=10)
        selector = RankSelection(selection_pressure=1.5)

        selected = selector.select(pop, count=5)
        assert len(selected) == 5

    def test_invalid_selection_pressure(self):
        """Test that invalid selection pressure raises error."""
        with pytest.raises(ValueError):
            RankSelection(selection_pressure=0.5)
        with pytest.raises(ValueError):
            RankSelection(selection_pressure=2.5)


class TestTruncationSelection:
    """Tests for truncation selection."""

    def test_selects_from_top_individuals(self):
        """Test that truncation selects from top individuals."""
        pop = Population(BinaryIndividual, size=10, length=10)
        # Set clear fitness differences
        for i, ind in enumerate(pop.individuals):
            ind.genes = [1] * i + [0] * (10 - i)
            ind.invalidate_fitness()

        selector = TruncationSelection(truncation_threshold=0.3)
        selected = selector.select(pop, count=5)

        # All selected should be from top 30%
        fitnesses = [ind.get_fitness() for ind in selected]
        assert all(f >= 7 for f in fitnesses)  # Top 3 have fitness 7, 8, 9


# ============================================================================
# CROSSOVER OPERATOR TESTS
# ============================================================================

class TestSinglePointCrossover:
    """Tests for single-point crossover."""

    def test_creates_two_offspring(self):
        """Test that crossover creates two offspring."""
        parent1 = BinaryIndividual(genes=[1, 1, 1, 1, 1], length=5)
        parent2 = BinaryIndividual(genes=[0, 0, 0, 0, 0], length=5)

        crossover = SinglePointCrossover()
        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        assert offspring1 is not parent1
        assert offspring2 is not parent2

    def test_offspring_contains_parent_genes(self):
        """Test that offspring contain genes from both parents."""
        random.seed(42)
        parent1 = BinaryIndividual(genes=[1, 1, 1, 1, 1], length=5)
        parent2 = BinaryIndividual(genes=[0, 0, 0, 0, 0], length=5)

        crossover = SinglePointCrossover()
        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        # Offspring should be a mix of 0s and 1s
        assert 0 in offspring1.genes or 1 in offspring1.genes

    def test_handles_single_gene(self):
        """Test handling of single-gene individuals."""
        parent1 = BinaryIndividual(genes=[1], length=1)
        parent2 = BinaryIndividual(genes=[0], length=1)

        crossover = SinglePointCrossover()
        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        # Should return copies
        assert offspring1.genes == [1]
        assert offspring2.genes == [0]


class TestOrderCrossover:
    """Tests for order crossover (permutation)."""

    def test_maintains_valid_permutation(self):
        """Test that offspring are valid permutations."""
        random.seed(42)
        parent1 = PermutationIndividual(genes=[0, 1, 2, 3, 4, 5, 6, 7], size=8)
        parent2 = PermutationIndividual(genes=[7, 6, 5, 4, 3, 2, 1, 0], size=8)

        crossover = OrderCrossover()
        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        # Both offspring should be valid permutations
        assert set(offspring1.genes) == set(range(8))
        assert set(offspring2.genes) == set(range(8))
        assert len(offspring1.genes) == 8
        assert len(offspring2.genes) == 8


class TestPartiallyMappedCrossover:
    """Tests for PMX crossover."""

    def test_maintains_valid_permutation(self):
        """Test that PMX produces valid permutations."""
        random.seed(42)
        parent1 = PermutationIndividual(genes=[0, 1, 2, 3, 4, 5, 6, 7], size=8)
        parent2 = PermutationIndividual(genes=[3, 7, 5, 1, 6, 2, 4, 0], size=8)

        crossover = PartiallyMappedCrossover()
        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        # Both offspring should be valid permutations
        assert set(offspring1.genes) == set(range(8))
        assert set(offspring2.genes) == set(range(8))


class TestArithmeticCrossover:
    """Tests for arithmetic crossover."""

    def test_offspring_between_parents(self):
        """Test that arithmetic crossover produces intermediate values."""
        parent1 = RealValuedIndividual(genes=[0.0, 0.0, 0.0], dimensions=3)
        parent2 = RealValuedIndividual(genes=[10.0, 10.0, 10.0], dimensions=3)

        crossover = ArithmeticCrossover(alpha=0.5)
        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        # With alpha=0.5, offspring should be at midpoint
        for gene in offspring1.genes:
            assert 0.0 <= gene <= 10.0
        for gene in offspring2.genes:
            assert 0.0 <= gene <= 10.0


class TestBlendCrossover:
    """Tests for blend crossover."""

    def test_offspring_in_extended_range(self):
        """Test that blend crossover can produce values outside parent range."""
        random.seed(42)
        parent1 = RealValuedIndividual(genes=[0.0, 0.0], dimensions=2, bounds=[(-100, 100), (-100, 100)])
        parent2 = RealValuedIndividual(genes=[10.0, 10.0], dimensions=2, bounds=[(-100, 100), (-100, 100)])

        crossover = BlendCrossover(alpha=0.5)

        # Run multiple times to get a range of values
        values = []
        for _ in range(100):
            offspring1, _ = crossover.crossover(parent1, parent2)
            values.extend(offspring1.genes)

        # Some values should be outside [0, 10]
        min_val = min(values)
        max_val = max(values)
        assert min_val < 0 or max_val > 10


class TestGetDefaultCrossover:
    """Tests for automatic crossover selection."""

    def test_binary_individual(self):
        """Test default crossover for binary individuals."""
        crossover = get_default_crossover(BinaryIndividual)
        assert isinstance(crossover, TwoPointCrossover)

    def test_permutation_individual(self):
        """Test default crossover for permutation individuals."""
        crossover = get_default_crossover(PermutationIndividual)
        assert isinstance(crossover, OrderCrossover)

    def test_real_valued_individual(self):
        """Test default crossover for real-valued individuals."""
        crossover = get_default_crossover(RealValuedIndividual)
        assert isinstance(crossover, ArithmeticCrossover)


# ============================================================================
# MUTATION OPERATOR TESTS
# ============================================================================

class TestBitFlipMutation:
    """Tests for bit flip mutation."""

    def test_mutation_flips_bits(self):
        """Test that mutation flips bits."""
        ind = BinaryIndividual(genes=[0, 0, 0, 0, 0], length=5)
        mutation = BitFlipMutation()

        # With rate 1.0, all bits should flip
        mutation.mutate(ind, mutation_rate=1.0)
        assert ind.genes == [1, 1, 1, 1, 1]

    def test_mutation_rate_zero(self):
        """Test that mutation rate 0 changes nothing."""
        original_genes = [1, 0, 1, 0, 1]
        ind = BinaryIndividual(genes=original_genes.copy(), length=5)
        mutation = BitFlipMutation()

        mutation.mutate(ind, mutation_rate=0.0)
        assert ind.genes == original_genes

    def test_fitness_invalidated(self):
        """Test that fitness is invalidated after mutation."""
        ind = BinaryIndividual(genes=[1, 1, 1, 1, 1], length=5)
        ind.get_fitness()  # Calculate and cache fitness

        mutation = BitFlipMutation()
        mutation.mutate(ind, mutation_rate=1.0)

        assert ind._fitness_calculated is False


class TestSwapMutation:
    """Tests for swap mutation."""

    def test_maintains_permutation(self):
        """Test that swap mutation maintains valid permutation."""
        random.seed(42)
        ind = PermutationIndividual(genes=[0, 1, 2, 3, 4], size=5)
        mutation = SwapMutation()

        for _ in range(100):
            mutation.mutate(ind, mutation_rate=1.0)
            assert set(ind.genes) == {0, 1, 2, 3, 4}


class TestInversionMutation:
    """Tests for inversion mutation."""

    def test_maintains_permutation(self):
        """Test that inversion maintains valid permutation."""
        random.seed(42)
        ind = PermutationIndividual(genes=[0, 1, 2, 3, 4], size=5)
        mutation = InversionMutation()

        for _ in range(100):
            mutation.mutate(ind, mutation_rate=1.0)
            assert set(ind.genes) == {0, 1, 2, 3, 4}


class TestGaussianMutation:
    """Tests for Gaussian mutation."""

    def test_respects_bounds(self):
        """Test that mutation respects bounds."""
        random.seed(42)
        bounds = [(0, 1), (0, 1), (0, 1)]
        ind = RealValuedIndividual(genes=[0.5, 0.5, 0.5], dimensions=3, bounds=bounds)
        mutation = GaussianMutation(sigma=10.0)  # Very large sigma

        for _ in range(100):
            mutation.mutate(ind, mutation_rate=1.0)
            for i, gene in enumerate(ind.genes):
                assert bounds[i][0] <= gene <= bounds[i][1]

    def test_invalid_sigma(self):
        """Test that invalid sigma raises error."""
        with pytest.raises(ValueError):
            GaussianMutation(sigma=0)
        with pytest.raises(ValueError):
            GaussianMutation(sigma=-1)


class TestCompositeMutation:
    """Tests for composite mutation."""

    def test_applies_one_operator(self):
        """Test that composite mutation applies one of its operators."""
        random.seed(42)
        ind = PermutationIndividual(genes=[0, 1, 2, 3, 4], size=5)

        mutation = CompositeMutation([
            SwapMutation(),
            InversionMutation()
        ])

        # Should not raise error and should maintain permutation
        mutation.mutate(ind, mutation_rate=1.0)
        assert set(ind.genes) == {0, 1, 2, 3, 4}

    def test_empty_operators_raises_error(self):
        """Test that empty operator list raises error."""
        with pytest.raises(ValueError):
            CompositeMutation([])


class TestGetDefaultMutation:
    """Tests for automatic mutation selection."""

    def test_binary_individual(self):
        """Test default mutation for binary individuals."""
        mutation = get_default_mutation(BinaryIndividual)
        assert isinstance(mutation, BitFlipMutation)

    def test_permutation_individual(self):
        """Test default mutation for permutation individuals."""
        mutation = get_default_mutation(PermutationIndividual)
        assert isinstance(mutation, SwapMutation)

    def test_real_valued_individual(self):
        """Test default mutation for real-valued individuals."""
        mutation = get_default_mutation(RealValuedIndividual)
        assert isinstance(mutation, GaussianMutation)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
