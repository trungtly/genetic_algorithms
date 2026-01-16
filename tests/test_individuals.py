"""
Unit tests for the Individual classes.
"""

import pytest
import random
from genetic_algorithms.core.individual import (
    Individual, BinaryIndividual, PermutationIndividual, RealValuedIndividual
)


class TestBinaryIndividual:
    """Tests for BinaryIndividual class."""

    def test_initialization_with_genes(self):
        """Test creating individual with provided genes."""
        genes = [1, 0, 1, 1, 0]
        ind = BinaryIndividual(genes=genes, length=5)
        assert ind.genes == genes
        assert len(ind.genes) == 5

    def test_initialization_random(self):
        """Test creating individual with random genes."""
        random.seed(42)
        ind = BinaryIndividual(length=10)
        assert len(ind.genes) == 10
        assert all(g in [0, 1] for g in ind.genes)

    def test_default_fitness_calculation(self):
        """Test default fitness (count of 1s)."""
        genes = [1, 1, 1, 0, 0]
        ind = BinaryIndividual(genes=genes, length=5)
        assert ind.calculate_fitness() == 3

    def test_fitness_caching(self):
        """Test that fitness is cached."""
        genes = [1, 1, 0, 0, 0]
        ind = BinaryIndividual(genes=genes, length=5)

        # First call calculates fitness
        fitness1 = ind.get_fitness()
        assert ind._fitness_calculated is True

        # Second call uses cached value
        fitness2 = ind.get_fitness()
        assert fitness1 == fitness2

    def test_fitness_invalidation(self):
        """Test fitness invalidation after mutation."""
        genes = [1, 1, 0, 0, 0]
        ind = BinaryIndividual(genes=genes, length=5)

        ind.get_fitness()
        assert ind._fitness_calculated is True

        ind.invalidate_fitness()
        assert ind._fitness_calculated is False
        assert ind.fitness is None

    def test_copy(self):
        """Test individual copying."""
        genes = [1, 0, 1, 0, 1]
        ind = BinaryIndividual(genes=genes, length=5)
        ind.age = 5
        ind.get_fitness()

        copy = ind.copy()

        # Should have same genes
        assert copy.genes == ind.genes
        # Should be independent
        assert copy.genes is not ind.genes
        # Age should be reset
        assert copy.age == 0

    def test_mutation(self):
        """Test bit flip mutation."""
        random.seed(42)
        genes = [0, 0, 0, 0, 0]
        ind = BinaryIndividual(genes=genes, length=5)

        # With mutation_rate=1.0, all bits should flip
        ind.mutate(mutation_rate=1.0)
        assert ind.genes == [1, 1, 1, 1, 1]

    def test_comparison(self):
        """Test individual comparison based on fitness."""
        ind1 = BinaryIndividual(genes=[1, 1, 1], length=3)
        ind2 = BinaryIndividual(genes=[1, 0, 0], length=3)

        assert ind1 > ind2  # 3 > 1
        assert not ind1 < ind2

    def test_equality(self):
        """Test individual equality based on genes."""
        ind1 = BinaryIndividual(genes=[1, 0, 1], length=3)
        ind2 = BinaryIndividual(genes=[1, 0, 1], length=3)
        ind3 = BinaryIndividual(genes=[0, 1, 0], length=3)

        assert ind1 == ind2
        assert ind1 != ind3


class TestPermutationIndividual:
    """Tests for PermutationIndividual class."""

    def test_initialization_with_genes(self):
        """Test creating individual with provided genes."""
        genes = [0, 2, 1, 3, 4]
        ind = PermutationIndividual(genes=genes, size=5)
        assert ind.genes == genes

    def test_initialization_random(self):
        """Test creating individual with random permutation."""
        random.seed(42)
        ind = PermutationIndividual(size=5)

        # Should be a valid permutation
        assert len(ind.genes) == 5
        assert set(ind.genes) == {0, 1, 2, 3, 4}

    def test_swap_mutation(self):
        """Test swap mutation maintains permutation validity."""
        random.seed(42)
        genes = [0, 1, 2, 3, 4]
        ind = PermutationIndividual(genes=genes.copy(), size=5)

        ind.mutate(mutation_rate=1.0)  # Guarantee mutation

        # Should still be a valid permutation
        assert set(ind.genes) == {0, 1, 2, 3, 4}

    def test_string_representation(self):
        """Test string representation."""
        genes = [0, 1, 2, 3, 4]
        ind = PermutationIndividual(genes=genes, size=5)
        str_repr = str(ind)
        assert "Individual" in str_repr


class TestRealValuedIndividual:
    """Tests for RealValuedIndividual class."""

    def test_initialization_with_genes(self):
        """Test creating individual with provided genes."""
        genes = [1.5, -2.0, 3.14]
        ind = RealValuedIndividual(genes=genes, dimensions=3)
        assert ind.genes == genes

    def test_initialization_random(self):
        """Test creating individual with random genes."""
        random.seed(42)
        bounds = [(0, 10), (-5, 5), (0, 100)]
        ind = RealValuedIndividual(dimensions=3, bounds=bounds)

        assert len(ind.genes) == 3
        for i, gene in enumerate(ind.genes):
            assert bounds[i][0] <= gene <= bounds[i][1]

    def test_default_fitness_sphere_function(self):
        """Test default fitness (negative sphere function)."""
        genes = [0.0, 0.0, 0.0]
        ind = RealValuedIndividual(genes=genes, dimensions=3)

        # At origin, sphere function = 0
        assert ind.calculate_fitness() == 0.0

    def test_gaussian_mutation(self):
        """Test Gaussian mutation respects bounds."""
        random.seed(42)
        bounds = [(0, 1), (0, 1)]
        genes = [0.5, 0.5]
        ind = RealValuedIndividual(genes=genes.copy(), dimensions=2, bounds=bounds)

        for _ in range(100):
            ind.mutate(mutation_rate=1.0)
            for i, gene in enumerate(ind.genes):
                assert bounds[i][0] <= gene <= bounds[i][1]


class TestIndividualAge:
    """Tests for individual age tracking."""

    def test_initial_age(self):
        """Test individuals start with age 0."""
        ind = BinaryIndividual(length=5)
        assert ind.age == 0

    def test_age_increment(self):
        """Test age increment."""
        ind = BinaryIndividual(length=5)
        ind.age_increment()
        assert ind.age == 1
        ind.age_increment()
        assert ind.age == 2

    def test_copy_resets_age(self):
        """Test that copying resets age."""
        ind = BinaryIndividual(length=5)
        ind.age = 10

        copy = ind.copy()
        assert copy.age == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
