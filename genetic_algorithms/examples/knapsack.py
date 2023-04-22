"""
Knapsack Problem Implementation using Genetic Algorithms

The 0/1 Knapsack Problem is a classic optimization problem:
- Given a set of items, each with a weight and value
- Find the combination of items that maximizes value
- Subject to the constraint that total weight ≤ knapsack capacity

This is an excellent introduction to genetic algorithms because:
- Simple binary representation (include item or not)
- Clear constraint handling
- Easy to understand and visualize
- Well-studied problem with known optimal solutions
"""

from typing import List, Tuple, Dict, Any
import random
import json
from ..core.individual import BinaryIndividual


class Item:
    """Represents an item that can be placed in the knapsack."""
    
    def __init__(self, weight: float, value: float, name: str = ""):
        """
        Initialize an item.
        
        Args:
            weight: Item weight (must be positive)
            value: Item value (must be non-negative)
            name: Optional item name for identification
        """
        if weight <= 0:
            raise ValueError("Item weight must be positive")
        if value < 0:
            raise ValueError("Item value must be non-negative")
        
        self.weight = weight
        self.value = value
        self.name = name or f"Item({weight}, {value})"
        
        # Value-to-weight ratio (useful for heuristics)
        self.efficiency = value / weight
    
    def __str__(self) -> str:
        return f"{self.name}: weight={self.weight}, value={self.value}, efficiency={self.efficiency:.3f}"
    
    def __repr__(self) -> str:
        return f"Item(weight={self.weight}, value={self.value}, name='{self.name}')"


class KnapsackIndividual(BinaryIndividual):
    """
    Individual representation for the knapsack problem.
    
    Uses binary encoding where each bit represents whether an item is included:
    - 1: Item is included in the knapsack
    - 0: Item is not included
    
    Example: [1, 0, 1, 0, 1] means items 0, 2, and 4 are selected
    """
    
    def __init__(self, genes: List[int] = None, items: List[Item] = None, 
                 capacity: float = 100.0):
        """
        Initialize a knapsack individual.
        
        Args:
            genes: Binary genes (0/1 for each item). If None, random genes generated.
            items: List of available items
            capacity: Knapsack capacity constraint
        """
        if items is None:
            raise ValueError("Items list is required for knapsack problem")
        
        self.items = items
        self.capacity = capacity
        
        # Initialize with proper length based on number of items
        super().__init__(genes, length=len(items))
        
        # Cache for efficiency
        self._total_weight = None
        self._total_value = None
        self._is_feasible = None
    
    def calculate_fitness(self) -> float:
        """
        Calculate fitness for this knapsack solution.
        
        Fitness function:
        - If solution is feasible (weight ≤ capacity): fitness = total_value
        - If infeasible (weight > capacity): apply penalty
        
        Returns:
            Fitness value (higher is better)
        """
        total_value = self.get_total_value()
        total_weight = self.get_total_weight()
        
        if total_weight <= self.capacity:
            # Feasible solution: fitness = value
            return total_value
        else:
            # Infeasible solution: apply penalty
            # Penalty is proportional to weight violation
            weight_violation = total_weight - self.capacity
            penalty_factor = weight_violation / self.capacity
            
            # Reduce fitness based on violation magnitude
            # Still give some credit for high-value solutions
            return total_value * (1.0 - penalty_factor)
    
    def get_total_weight(self) -> float:
        """Get total weight of selected items."""
        if self._total_weight is None:
            self._total_weight = sum(
                self.items[i].weight for i in range(len(self.genes))
                if self.genes[i] == 1
            )
        return self._total_weight
    
    def get_total_value(self) -> float:
        """Get total value of selected items."""
        if self._total_value is None:
            self._total_value = sum(
                self.items[i].value for i in range(len(self.genes))
                if self.genes[i] == 1
            )
        return self._total_value
    
    def is_feasible(self) -> bool:
        """Check if solution satisfies weight constraint."""
        if self._is_feasible is None:
            self._is_feasible = self.get_total_weight() <= self.capacity
        return self._is_feasible
    
    def get_selected_items(self) -> List[Item]:
        """Get list of items selected in this solution."""
        return [self.items[i] for i in range(len(self.genes)) if self.genes[i] == 1]
    
    def get_solution_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of this solution."""
        selected_items = self.get_selected_items()
        return {
            'total_value': self.get_total_value(),
            'total_weight': self.get_total_weight(),
            'capacity': self.capacity,
            'is_feasible': self.is_feasible(),
            'weight_utilization': self.get_total_weight() / self.capacity,
            'num_items_selected': sum(self.genes),
            'num_items_available': len(self.genes),
            'selected_items': [str(item) for item in selected_items],
            'fitness': self.get_fitness()
        }
    
    def invalidate_fitness(self):
        """Invalidate cached values when genes change."""
        super().invalidate_fitness()
        self._total_weight = None
        self._total_value = None
        self._is_feasible = None
    
    def repair(self) -> 'KnapsackIndividual':
        """
        Repair infeasible solution using greedy removal.
        
        If the solution violates weight constraint, remove items
        with lowest value-to-weight ratio until feasible.
        
        Returns:
            Repaired individual (self, modified in place)
        """
        if self.is_feasible():
            return self  # Already feasible
        
        # Get selected items with their indices
        selected_items_with_indices = [
            (i, self.items[i]) for i in range(len(self.genes))
            if self.genes[i] == 1
        ]
        
        # Sort by efficiency (value/weight ratio) - remove least efficient first
        selected_items_with_indices.sort(key=lambda x: x[1].efficiency)
        
        # Remove items until feasible
        for idx, item in selected_items_with_indices:
            if self.get_total_weight() <= self.capacity:
                break
            
            # Remove this item
            self.genes[idx] = 0
            self.invalidate_fitness()
        
        return self
    
    def improve_with_greedy(self) -> 'KnapsackIndividual':
        """
        Improve solution by greedily adding high-efficiency items.
        
        Try to add items with highest value-to-weight ratio that fit.
        
        Returns:
            Improved individual (self, modified in place)
        """
        # Get unselected items with their indices
        unselected_items_with_indices = [
            (i, self.items[i]) for i in range(len(self.genes))
            if self.genes[i] == 0
        ]
        
        # Sort by efficiency (highest first)
        unselected_items_with_indices.sort(key=lambda x: x[1].efficiency, reverse=True)
        
        # Try to add items
        for idx, item in unselected_items_with_indices:
            if self.get_total_weight() + item.weight <= self.capacity:
                # Add this item
                self.genes[idx] = 1
                self.invalidate_fitness()
        
        return self
    
    def __str__(self) -> str:
        """String representation of the knapsack solution."""
        return (f"KnapsackSolution(value={self.get_total_value():.1f}, "
                f"weight={self.get_total_weight():.1f}/{self.capacity:.1f}, "
                f"feasible={self.is_feasible()}, items={sum(self.genes)}/{len(self.genes)})")


class KnapsackProblem:
    """
    Knapsack problem instance with utilities for problem generation and analysis.
    
    This class provides:
    - Problem instance creation (random or from data)
    - Optimal solution computation (for small instances)
    - Solution analysis and comparison
    - Problem difficulty assessment
    """
    
    def __init__(self, items: List[Item], capacity: float):
        """
        Initialize knapsack problem instance.
        
        Args:
            items: List of available items
            capacity: Knapsack capacity constraint
        """
        if not items:
            raise ValueError("At least one item is required")
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.items = items
        self.capacity = capacity
        self.num_items = len(items)
        
        # Problem statistics
        self.total_weight = sum(item.weight for item in items)
        self.total_value = sum(item.value for item in items)
        self.avg_efficiency = sum(item.efficiency for item in items) / len(items)
    
    def create_individual(self, genes: List[int] = None) -> KnapsackIndividual:
        """Create a knapsack individual for this problem."""
        return KnapsackIndividual(genes=genes, items=self.items, capacity=self.capacity)
    
    def solve_with_ga(self, **ga_params) -> Tuple[KnapsackIndividual, Dict[str, Any]]:
        """
        Solve this knapsack problem using genetic algorithm.
        
        Args:
            **ga_params: Parameters for genetic algorithm
            
        Returns:
            Tuple of (best_solution, statistics)
        """
        from ..core import solve_with_ga
        
        # Default parameters optimized for knapsack problem
        default_params = {
            'population_size': 100,
            'max_generations': 500,
            'mutation_rate': 1.0 / self.num_items,  # About 1 bit per individual
            'crossover_rate': 0.8,
            'elite_count': 2
        }
        default_params.update(ga_params)
        
        return solve_with_ga(
            individual_class=KnapsackIndividual,
            items=self.items,
            capacity=self.capacity,
            **default_params
        )
    
    def greedy_solution(self) -> KnapsackIndividual:
        """
        Solve using greedy heuristic (sort by efficiency, add until capacity).
        
        This provides a good baseline and often finds high-quality solutions
        quickly for the knapsack problem.
        
        Returns:
            Greedy solution
        """
        # Sort items by efficiency (value/weight ratio)
        items_with_indices = [(i, item) for i, item in enumerate(self.items)]
        items_with_indices.sort(key=lambda x: x[1].efficiency, reverse=True)
        
        # Greedy selection
        genes = [0] * self.num_items
        current_weight = 0
        
        for idx, item in items_with_indices:
            if current_weight + item.weight <= self.capacity:
                genes[idx] = 1
                current_weight += item.weight
        
        return self.create_individual(genes)
    
    def random_solution(self) -> KnapsackIndividual:
        """Generate a random solution."""
        return self.create_individual()  # Random genes generated automatically
    
    def brute_force_optimal(self) -> KnapsackIndividual:
        """
        Find optimal solution using brute force enumeration.
        
        WARNING: Only use for small problems (< 20 items) as complexity is O(2^n).
        
        Returns:
            Optimal solution
        """
        if self.num_items > 20:
            raise ValueError(f"Too many items ({self.num_items}) for brute force. Use <= 20 items.")
        
        best_solution = None
        best_fitness = -1
        
        # Try all possible combinations (2^n)
        for i in range(2 ** self.num_items):
            # Convert integer to binary representation
            genes = [(i >> j) & 1 for j in range(self.num_items)]
            
            # Create and evaluate solution
            solution = self.create_individual(genes)
            if solution.is_feasible():
                fitness = solution.get_fitness()
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution
        
        return best_solution
    
    def analyze_difficulty(self) -> Dict[str, Any]:
        """
        Analyze problem difficulty characteristics.
        
        Returns:
            Dictionary with difficulty metrics
        """
        # Calculate tightness ratio (how constraining the capacity is)
        min_weight = min(item.weight for item in self.items)
        max_weight = max(item.weight for item in self.items)
        weight_range = max_weight - min_weight
        tightness_ratio = self.capacity / self.total_weight
        
        # Calculate efficiency variance (how different item efficiencies are)
        efficiencies = [item.efficiency for item in self.items]
        efficiency_variance = sum((e - self.avg_efficiency) ** 2 for e in efficiencies) / len(efficiencies)
        
        # Estimate problem difficulty
        difficulty_score = 0
        if tightness_ratio < 0.3:
            difficulty_score += 3  # Very tight capacity
        elif tightness_ratio < 0.6:
            difficulty_score += 2  # Moderately tight
        else:
            difficulty_score += 1  # Loose capacity
        
        if efficiency_variance > self.avg_efficiency * 0.5:
            difficulty_score += 2  # High efficiency variance
        else:
            difficulty_score += 1  # Low efficiency variance
        
        difficulty_levels = {
            2: "Easy",
            3: "Medium", 
            4: "Hard",
            5: "Very Hard"
        }
        
        return {
            'num_items': self.num_items,
            'capacity': self.capacity,
            'total_weight': self.total_weight,
            'total_value': self.total_value,
            'tightness_ratio': tightness_ratio,
            'avg_efficiency': self.avg_efficiency,
            'efficiency_variance': efficiency_variance,
            'weight_range': weight_range,
            'difficulty_score': difficulty_score,
            'difficulty_level': difficulty_levels.get(difficulty_score, "Extreme")
        }
    
    def get_problem_summary(self) -> str:
        """Get human-readable problem summary."""
        analysis = self.analyze_difficulty()
        
        summary = f"""
Knapsack Problem Summary:
========================
Items: {self.num_items}
Capacity: {self.capacity:.1f}
Total Weight: {self.total_weight:.1f} (Tightness: {analysis['tightness_ratio']:.2f})
Total Value: {self.total_value:.1f}
Average Efficiency: {self.avg_efficiency:.3f}
Difficulty: {analysis['difficulty_level']} (Score: {analysis['difficulty_score']}/5)

Items:
------"""
        
        for i, item in enumerate(self.items):
            summary += f"\n{i:2d}: {item}"
        
        return summary
    
    @classmethod
    def create_random_problem(cls, num_items: int, capacity: float = None,
                            weight_range: Tuple[float, float] = (1, 20),
                            value_range: Tuple[float, float] = (1, 100),
                            correlation: float = 0.0) -> 'KnapsackProblem':
        """
        Create a random knapsack problem instance.
        
        Args:
            num_items: Number of items to generate
            capacity: Knapsack capacity (if None, set to ~60% of total weight)
            weight_range: (min_weight, max_weight) for items
            value_range: (min_value, max_value) for items  
            correlation: Weight-value correlation (-1 to 1)
                        0 = no correlation, 1 = higher weight → higher value
            
        Returns:
            Random knapsack problem instance
        """
        items = []
        
        for i in range(num_items):
            # Generate weight
            weight = random.uniform(weight_range[0], weight_range[1])
            
            # Generate value with optional correlation to weight
            if correlation == 0.0:
                # No correlation - random value
                value = random.uniform(value_range[0], value_range[1])
            else:
                # Correlated value
                weight_normalized = (weight - weight_range[0]) / (weight_range[1] - weight_range[0])
                
                if correlation > 0:
                    # Positive correlation: higher weight → higher value tendency
                    base_value = value_range[0] + weight_normalized * (value_range[1] - value_range[0])
                    noise = random.uniform(-1, 1) * (1 - correlation) * (value_range[1] - value_range[0]) * 0.5
                    value = max(value_range[0], min(value_range[1], base_value + noise))
                else:
                    # Negative correlation: higher weight → lower value tendency
                    base_value = value_range[1] - weight_normalized * (value_range[1] - value_range[0])
                    noise = random.uniform(-1, 1) * (1 + correlation) * (value_range[1] - value_range[0]) * 0.5
                    value = max(value_range[0], min(value_range[1], base_value + noise))
            
            items.append(Item(weight=weight, value=value, name=f"Item_{i}"))
        
        # Set capacity if not provided
        if capacity is None:
            total_weight = sum(item.weight for item in items)
            capacity = total_weight * 0.6  # 60% of total weight
        
        return cls(items, capacity)
    
    @classmethod
    def create_standard_problem(cls, problem_name: str) -> 'KnapsackProblem':
        """
        Create a standard benchmark knapsack problem.
        
        Args:
            problem_name: Name of standard problem ('small', 'medium', 'large')
            
        Returns:
            Standard problem instance
        """
        if problem_name == 'small':
            # Small problem for testing and education
            items = [
                Item(10, 60, "Gold"),
                Item(20, 100, "Silver"), 
                Item(30, 120, "Bronze"),
                Item(40, 160, "Platinum"),
                Item(5, 30, "Diamond"),
                Item(15, 90, "Ruby"),
                Item(25, 150, "Emerald")
            ]
            capacity = 80
            
        elif problem_name == 'medium':
            # Medium problem with 15 items
            random.seed(42)  # For reproducibility
            return cls.create_random_problem(
                num_items=15,
                capacity=100,
                weight_range=(5, 25),
                value_range=(10, 80),
                correlation=0.3
            )
            
        elif problem_name == 'large':
            # Large problem with 50 items
            random.seed(123)  # For reproducibility  
            return cls.create_random_problem(
                num_items=50,
                capacity=200,
                weight_range=(1, 30),
                value_range=(1, 100),
                correlation=0.1
            )
            
        else:
            raise ValueError(f"Unknown standard problem: {problem_name}")
        
        return cls(items, capacity)
    
    def save_to_file(self, filename: str):
        """Save problem instance to JSON file."""
        data = {
            'capacity': self.capacity,
            'items': [
                {
                    'weight': item.weight,
                    'value': item.value,
                    'name': item.name
                }
                for item in self.items
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'KnapsackProblem':
        """Load problem instance from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        items = [
            Item(
                weight=item_data['weight'],
                value=item_data['value'],
                name=item_data.get('name', f"Item_{i}")
            )
            for i, item_data in enumerate(data['items'])
        ]
        
        return cls(items, data['capacity'])