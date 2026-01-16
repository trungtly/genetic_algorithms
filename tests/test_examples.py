"""
Unit tests for example problems (TSP and Knapsack).
"""

import pytest
import random
from genetic_algorithms.examples.tsp import TSPProblem, TSPIndividual, City
from genetic_algorithms.examples.knapsack import KnapsackProblem, KnapsackIndividual, Item


# ============================================================================
# TSP TESTS
# ============================================================================

class TestCity:
    """Tests for City class."""

    def test_initialization(self):
        """Test city initialization."""
        city = City(10.0, 20.0, "TestCity")
        assert city.x == 10.0
        assert city.y == 20.0
        assert city.name == "TestCity"

    def test_distance_calculation(self):
        """Test distance calculation between cities."""
        city1 = City(0.0, 0.0)
        city2 = City(3.0, 4.0)

        distance = city1.distance_to(city2)
        assert abs(distance - 5.0) < 1e-6  # Should be 5 (3-4-5 triangle)

    def test_distance_symmetry(self):
        """Test that distance is symmetric."""
        city1 = City(1.0, 2.0)
        city2 = City(5.0, 7.0)

        assert abs(city1.distance_to(city2) - city2.distance_to(city1)) < 1e-6


class TestTSPIndividual:
    """Tests for TSPIndividual class."""

    def test_initialization(self):
        """Test TSP individual initialization."""
        cities = [City(0, 0), City(1, 0), City(1, 1), City(0, 1)]
        ind = TSPIndividual(genes=[0, 1, 2, 3], cities=cities)

        assert len(ind.genes) == 4
        assert set(ind.genes) == {0, 1, 2, 3}

    def test_requires_cities(self):
        """Test that cities are required."""
        with pytest.raises(ValueError):
            TSPIndividual(genes=[0, 1, 2])

    def test_random_initialization(self):
        """Test random tour generation."""
        random.seed(42)
        cities = [City(i, i) for i in range(5)]
        ind = TSPIndividual(cities=cities)

        assert len(ind.genes) == 5
        assert set(ind.genes) == {0, 1, 2, 3, 4}

    def test_total_distance(self):
        """Test total distance calculation."""
        # Square: 0,0 -> 1,0 -> 1,1 -> 0,1 -> back
        cities = [City(0, 0), City(1, 0), City(1, 1), City(0, 1)]
        ind = TSPIndividual(genes=[0, 1, 2, 3], cities=cities)

        # Perimeter of unit square = 4
        assert abs(ind.get_total_distance() - 4.0) < 1e-6

    def test_fitness_is_negative_distance(self):
        """Test that fitness is negative distance."""
        cities = [City(0, 0), City(1, 0), City(1, 1), City(0, 1)]
        ind = TSPIndividual(genes=[0, 1, 2, 3], cities=cities)

        fitness = ind.get_fitness()
        distance = ind.get_total_distance()

        # Fitness should be approximately -distance
        assert fitness < 0
        assert abs(fitness + distance) < 1e-6

    def test_2opt_improvement(self):
        """Test 2-opt local search improvement."""
        random.seed(42)
        cities = [City(0, 0), City(1, 0), City(2, 0), City(2, 1), City(1, 1), City(0, 1)]

        # Create a suboptimal tour
        ind = TSPIndividual(genes=[0, 2, 1, 3, 5, 4], cities=cities)
        original_distance = ind.get_total_distance()

        ind.apply_2opt_improvement()
        improved_distance = ind.get_total_distance()

        # 2-opt should improve or maintain distance
        assert improved_distance <= original_distance


class TestTSPProblem:
    """Tests for TSPProblem class."""

    def test_initialization(self):
        """Test problem initialization."""
        cities = [City(0, 0), City(1, 0), City(1, 1)]
        problem = TSPProblem(cities)

        assert problem.num_cities == 3

    def test_requires_minimum_cities(self):
        """Test that at least 3 cities are required."""
        with pytest.raises(ValueError):
            TSPProblem([City(0, 0), City(1, 1)])

    def test_random_problem_creation(self):
        """Test random problem generation."""
        random.seed(42)
        problem = TSPProblem.create_random_problem(10)

        assert problem.num_cities == 10

    def test_circle_problem(self):
        """Test circle problem generation."""
        problem = TSPProblem.create_circle_problem(8)

        assert problem.num_cities == 8

    def test_grid_problem(self):
        """Test grid problem generation."""
        problem = TSPProblem.create_grid_problem(3, 3)

        assert problem.num_cities == 9

    def test_nearest_neighbor_solution(self):
        """Test nearest neighbor heuristic."""
        cities = [City(0, 0), City(1, 0), City(2, 0), City(3, 0)]
        problem = TSPProblem(cities)

        solution = problem.nearest_neighbor_solution(start_city=0)

        # Should produce a valid tour
        assert set(solution.genes) == {0, 1, 2, 3}

    def test_brute_force_optimal_small(self):
        """Test brute force on small problem."""
        # Create a simple square
        cities = [City(0, 0), City(1, 0), City(1, 1), City(0, 1)]
        problem = TSPProblem(cities)

        optimal = problem.brute_force_optimal()

        # Optimal tour of unit square should have distance 4
        assert abs(optimal.get_total_distance() - 4.0) < 1e-6

    def test_brute_force_too_large(self):
        """Test that brute force rejects large problems."""
        cities = [City(i, i) for i in range(15)]
        problem = TSPProblem(cities)

        with pytest.raises(ValueError):
            problem.brute_force_optimal()


# ============================================================================
# KNAPSACK TESTS
# ============================================================================

class TestItem:
    """Tests for Item class."""

    def test_initialization(self):
        """Test item initialization."""
        item = Item(weight=10, value=50, name="Gold")

        assert item.weight == 10
        assert item.value == 50
        assert item.name == "Gold"
        assert item.efficiency == 5.0

    def test_invalid_weight(self):
        """Test that invalid weight raises error."""
        with pytest.raises(ValueError):
            Item(weight=0, value=10)
        with pytest.raises(ValueError):
            Item(weight=-5, value=10)

    def test_invalid_value(self):
        """Test that negative value raises error."""
        with pytest.raises(ValueError):
            Item(weight=10, value=-5)


class TestKnapsackIndividual:
    """Tests for KnapsackIndividual class."""

    def test_initialization(self):
        """Test knapsack individual initialization."""
        items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        ind = KnapsackIndividual(genes=[1, 0, 1], items=items, capacity=50)

        assert len(ind.genes) == 3

    def test_requires_items(self):
        """Test that items are required."""
        with pytest.raises(ValueError):
            KnapsackIndividual(genes=[1, 0, 1], capacity=50)

    def test_total_weight(self):
        """Test total weight calculation."""
        items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        ind = KnapsackIndividual(genes=[1, 0, 1], items=items, capacity=100)

        # Weight: 10 + 30 = 40
        assert ind.get_total_weight() == 40

    def test_total_value(self):
        """Test total value calculation."""
        items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        ind = KnapsackIndividual(genes=[1, 0, 1], items=items, capacity=100)

        # Value: 60 + 120 = 180
        assert ind.get_total_value() == 180

    def test_feasibility(self):
        """Test feasibility check."""
        items = [Item(10, 60), Item(20, 100), Item(30, 120)]

        feasible = KnapsackIndividual(genes=[1, 1, 0], items=items, capacity=50)
        infeasible = KnapsackIndividual(genes=[1, 1, 1], items=items, capacity=50)

        assert feasible.is_feasible() is True
        assert infeasible.is_feasible() is False

    def test_fitness_feasible(self):
        """Test fitness for feasible solution."""
        items = [Item(10, 60), Item(20, 100)]
        ind = KnapsackIndividual(genes=[1, 1], items=items, capacity=50)

        # Feasible: fitness = total value = 160
        assert ind.get_fitness() == 160

    def test_fitness_infeasible_penalty(self):
        """Test fitness penalty for infeasible solution."""
        items = [Item(10, 60), Item(20, 100)]
        ind = KnapsackIndividual(genes=[1, 1], items=items, capacity=20)

        # Infeasible: fitness should be penalized
        assert ind.get_fitness() < 160

    def test_repair(self):
        """Test repair function."""
        items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        ind = KnapsackIndividual(genes=[1, 1, 1], items=items, capacity=40)

        assert ind.is_feasible() is False

        ind.repair()

        assert ind.is_feasible() is True

    def test_greedy_improvement(self):
        """Test greedy improvement."""
        items = [Item(10, 60), Item(5, 40), Item(15, 50)]
        ind = KnapsackIndividual(genes=[0, 0, 0], items=items, capacity=20)

        original_value = ind.get_total_value()
        ind.improve_with_greedy()

        # Should add items
        assert ind.get_total_value() > original_value
        assert ind.is_feasible()


class TestKnapsackProblem:
    """Tests for KnapsackProblem class."""

    def test_initialization(self):
        """Test problem initialization."""
        items = [Item(10, 60), Item(20, 100)]
        problem = KnapsackProblem(items, capacity=50)

        assert problem.num_items == 2
        assert problem.capacity == 50

    def test_requires_items(self):
        """Test that items are required."""
        with pytest.raises(ValueError):
            KnapsackProblem([], capacity=50)

    def test_requires_positive_capacity(self):
        """Test that capacity must be positive."""
        items = [Item(10, 60)]
        with pytest.raises(ValueError):
            KnapsackProblem(items, capacity=0)

    def test_random_problem_creation(self):
        """Test random problem generation."""
        random.seed(42)
        problem = KnapsackProblem.create_random_problem(10, capacity=100)

        assert problem.num_items == 10
        assert problem.capacity == 100

    def test_greedy_solution(self):
        """Test greedy heuristic."""
        items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        problem = KnapsackProblem(items, capacity=50)

        solution = problem.greedy_solution()

        assert solution.is_feasible()

    def test_brute_force_optimal(self):
        """Test brute force optimal solution."""
        items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        problem = KnapsackProblem(items, capacity=50)

        optimal = problem.brute_force_optimal()

        assert optimal.is_feasible()
        # Optimal is items 0 and 1: weight=30, value=160
        assert optimal.get_total_value() == 160

    def test_brute_force_too_large(self):
        """Test that brute force rejects large problems."""
        items = [Item(i+1, (i+1)*10) for i in range(25)]
        problem = KnapsackProblem(items, capacity=100)

        with pytest.raises(ValueError):
            problem.brute_force_optimal()

    def test_standard_problem_small(self):
        """Test standard small problem."""
        problem = KnapsackProblem.create_standard_problem('small')

        assert problem.num_items == 7
        assert problem.capacity == 80


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
