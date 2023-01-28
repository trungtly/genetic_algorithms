# Implementing Your Own Genetic Algorithm

This guide will walk you through implementing a genetic algorithm for your own problem using this framework. We'll cover everything from defining individuals to running the complete algorithm.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Defining Your Individual Class](#defining-your-individual-class)
3. [Setting Up the Problem](#setting-up-the-problem)
4. [Configuring the Genetic Algorithm](#configuring-the-genetic-algorithm)
5. [Running and Monitoring](#running-and-monitoring)
6. [Customizing Operators](#customizing-operators)
7. [Complete Examples](#complete-examples)

## Quick Start

The fastest way to get started is to use one of the built-in individual types:

```python
from genetic_algorithms.core import GeneticAlgorithm, BinaryIndividual

# For binary optimization problems
ga = GeneticAlgorithm(
    individual_class=BinaryIndividual,
    population_size=100,
    generations=50,
    chromosome_length=20,  # Required for BinaryIndividual
    fitness_function=your_fitness_function
)

best_individual, stats = ga.evolve()
print(f"Best solution: {best_individual.genes}")
print(f"Best fitness: {best_individual.fitness}")
```

## Defining Your Individual Class

For custom problems, you'll need to create your own Individual class. Here's the basic structure:

### Basic Individual Template

```python
from genetic_algorithms.core.individual import Individual
from typing import List, Any
import random

class MyIndividual(Individual):
    def __init__(self, genes: List[Any] = None, **kwargs):
        """Initialize your individual."""
        # Extract problem-specific data from kwargs
        self.problem_data = kwargs.get('problem_data', {})
        
        # Generate random genes if not provided
        if genes is None:
            genes = self.generate_random_genes()
        
        super().__init__(genes)
    
    def generate_random_genes(self) -> List[Any]:
        """Generate random genes for initialization."""
        # Implement random gene generation
        # Example for integer genes:
        return [random.randint(0, 100) for _ in range(10)]
    
    def calculate_fitness(self) -> float:
        """Calculate fitness for this individual."""
        # Implement your fitness calculation
        # Higher values = better fitness
        return sum(self.genes)  # Simple example
    
    def mutate(self, mutation_rate: float) -> 'MyIndividual':
        """Mutate this individual."""
        new_genes = self.genes.copy()
        
        for i in range(len(new_genes)):
            if random.random() < mutation_rate:
                # Implement mutation logic
                new_genes[i] = random.randint(0, 100)
        
        return MyIndividual(new_genes, problem_data=self.problem_data)
    
    def crossover(self, other: 'MyIndividual') -> tuple['MyIndividual', 'MyIndividual']:
        """Perform crossover with another individual."""
        # Example: single-point crossover
        crossover_point = random.randint(1, len(self.genes) - 1)
        
        child1_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        child2_genes = other.genes[:crossover_point] + self.genes[crossover_point:]
        
        child1 = MyIndividual(child1_genes, problem_data=self.problem_data)
        child2 = MyIndividual(child2_genes, problem_data=self.problem_data)
        
        return child1, child2
```

### Real-World Example: Function Optimization

Let's implement an individual for optimizing a mathematical function:

```python
import numpy as np
from genetic_algorithms.core.individual import Individual

class FunctionOptimizationIndividual(Individual):
    """Individual for optimizing continuous functions."""
    
    def __init__(self, genes: List[float] = None, **kwargs):
        self.bounds = kwargs.get('problem_data', {}).get('bounds', [(-10, 10)] * 5)
        self.function = kwargs.get('problem_data', {}).get('function')
        
        if genes is None:
            genes = self.generate_random_genes()
        
        super().__init__(genes)
    
    def generate_random_genes(self) -> List[float]:
        """Generate random real-valued genes within bounds."""
        return [
            np.random.uniform(low, high) 
            for low, high in self.bounds
        ]
    
    def calculate_fitness(self) -> float:
        """Evaluate the function at this point."""
        if self.function is None:
            raise ValueError("No function provided in problem_data")
        
        try:
            return self.function(self.genes)
        except Exception as e:
            # Return very poor fitness for invalid solutions
            return float('-inf')
    
    def mutate(self, mutation_rate: float) -> 'FunctionOptimizationIndividual':
        """Gaussian mutation with boundary constraints."""
        new_genes = []
        
        for i, gene in enumerate(self.genes):
            if np.random.random() < mutation_rate:
                # Gaussian mutation
                sigma = (self.bounds[i][1] - self.bounds[i][0]) * 0.1
                mutated_gene = gene + np.random.normal(0, sigma)
                
                # Ensure bounds are respected
                low, high = self.bounds[i]
                mutated_gene = max(low, min(high, mutated_gene))
                
                new_genes.append(mutated_gene)
            else:
                new_genes.append(gene)
        
        return FunctionOptimizationIndividual(
            new_genes, 
            problem_data={'bounds': self.bounds, 'function': self.function}
        )
    
    def crossover(self, other: 'FunctionOptimizationIndividual') -> tuple:
        """Arithmetic crossover for real-valued genes."""
        alpha = np.random.random()
        
        child1_genes = [
            alpha * self.genes[i] + (1 - alpha) * other.genes[i]
            for i in range(len(self.genes))
        ]
        
        child2_genes = [
            (1 - alpha) * self.genes[i] + alpha * other.genes[i]
            for i in range(len(self.genes))
        ]
        
        problem_data = {'bounds': self.bounds, 'function': self.function}
        
        child1 = FunctionOptimizationIndividual(child1_genes, problem_data=problem_data)
        child2 = FunctionOptimizationIndividual(child2_genes, problem_data=problem_data)
        
        return child1, child2
```

## Setting Up the Problem

### Method 1: Using problem_data Parameter

```python
# Define your problem
def rosenbrock_function(x):
    """Rosenbrock function - global minimum at (1, 1, ..., 1)."""
    return -sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                for i in range(len(x) - 1))

problem_data = {
    'function': rosenbrock_function,
    'bounds': [(-5, 5)] * 5  # 5-dimensional optimization
}

ga = GeneticAlgorithm(
    individual_class=FunctionOptimizationIndividual,
    population_size=100,
    generations=200,
    problem_data=problem_data
)
```

### Method 2: Using fitness_function Parameter

```python
# For simpler cases, use fitness_function directly
def simple_fitness(individual):
    # Access genes and calculate fitness
    return sum(gene**2 for gene in individual.genes)

ga = GeneticAlgorithm(
    individual_class=RealValuedIndividual,
    population_size=50,
    generations=100,
    fitness_function=simple_fitness,
    chromosome_length=10,
    gene_bounds=(-10, 10)
)
```

## Configuring the Genetic Algorithm

### Basic Configuration

```python
ga = GeneticAlgorithm(
    individual_class=YourIndividualClass,
    population_size=100,         # Size of population
    generations=200,             # Number of generations
    crossover_rate=0.8,          # Probability of crossover
    mutation_rate=0.1,           # Probability of mutation
    selection_method='tournament', # Selection method
    elitism=True,                # Keep best individuals
    elite_size=5                 # Number of elite individuals
)
```

### Advanced Configuration

```python
ga = GeneticAlgorithm(
    individual_class=YourIndividualClass,
    population_size=200,
    generations=500,
    
    # Selection parameters
    selection_method='tournament',
    tournament_size=5,           # For tournament selection
    
    # Crossover parameters
    crossover_rate=0.9,
    crossover_method='uniform',  # If custom crossover implemented
    
    # Mutation parameters
    mutation_rate=0.05,
    adaptive_mutation=True,      # Adapt mutation rate over time
    
    # Termination criteria
    target_fitness=100.0,        # Stop if this fitness is reached
    stagnation_generations=50,   # Stop if no improvement for N generations
    
    # Other parameters
    elitism=True,
    elite_size=10,
    verbose=True,                # Print progress
    random_seed=42               # For reproducible results
)
```

## Running and Monitoring

### Basic Execution

```python
# Run the algorithm
best_individual, stats_history = ga.evolve()

# Access results
print(f"Best fitness: {best_individual.fitness}")
print(f"Best genes: {best_individual.genes}")
print(f"Generations run: {len(stats_history)}")
```

### Real-time Monitoring

```python
# Custom callback for monitoring
def monitor_progress(generation, population, stats):
    if generation % 10 == 0:
        print(f"Generation {generation}: Best = {stats['best_fitness']:.4f}, "
              f"Avg = {stats['avg_fitness']:.4f}")

ga = GeneticAlgorithm(
    individual_class=YourIndividualClass,
    population_size=100,
    generations=200,
    callback=monitor_progress
)

best_individual, stats_history = ga.evolve()
```

### Using Statistics

```python
# Analyze the evolution statistics
for gen, stats in enumerate(stats_history):
    print(f"Generation {gen}:")
    print(f"  Best fitness: {stats['best_fitness']}")
    print(f"  Average fitness: {stats['avg_fitness']}")
    print(f"  Worst fitness: {stats['worst_fitness']}")
    print(f"  Diversity: {stats.get('diversity', 'N/A')}")
```

## Customizing Operators

### Custom Selection

```python
from genetic_algorithms.core.selection import Selection

class CustomSelection(Selection):
    @staticmethod
    def select(population, selection_size, **kwargs):
        # Implement your selection logic
        # Return selected individuals
        pass

# Use in GA
ga = GeneticAlgorithm(
    individual_class=YourIndividualClass,
    population_size=100,
    generations=200,
    selection_class=CustomSelection
)
```

### Custom Mutation Strategy

```python
class AdvancedIndividual(Individual):
    def mutate(self, mutation_rate: float) -> 'AdvancedIndividual':
        # Implement sophisticated mutation
        # - Adaptive mutation rates
        # - Multiple mutation operators
        # - Problem-specific mutations
        pass
```

## Complete Examples

### Example 1: Optimizing Himmelblau's Function

```python
import numpy as np
from genetic_algorithms.core import GeneticAlgorithm

def himmelblau_function(x):
    """Himmelblau's function with 4 global maxima."""
    return -((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)

problem_data = {
    'function': himmelblau_function,
    'bounds': [(-5, 5), (-5, 5)]
}

ga = GeneticAlgorithm(
    individual_class=FunctionOptimizationIndividual,
    population_size=100,
    generations=200,
    crossover_rate=0.8,
    mutation_rate=0.1,
    problem_data=problem_data,
    verbose=True
)

best_individual, stats_history = ga.evolve()

print(f"Best solution: {best_individual.genes}")
print(f"Best fitness: {best_individual.fitness}")

# Known global optima: (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
```

### Example 2: Feature Selection Problem

```python
from genetic_algorithms.core.individual import Individual
import numpy as np

class FeatureSelectionIndividual(Individual):
    """Individual for feature selection problems."""
    
    def __init__(self, genes: List[bool] = None, **kwargs):
        self.X_train = kwargs.get('problem_data', {}).get('X_train')
        self.y_train = kwargs.get('problem_data', {}).get('y_train')
        self.X_val = kwargs.get('problem_data', {}).get('X_val')
        self.y_val = kwargs.get('problem_data', {}).get('y_val')
        self.n_features = kwargs.get('problem_data', {}).get('n_features')
        
        if genes is None:
            genes = self.generate_random_genes()
        
        super().__init__(genes)
    
    def generate_random_genes(self) -> List[bool]:
        # Start with some features selected
        genes = [False] * self.n_features
        n_selected = np.random.randint(1, max(2, self.n_features // 4))
        selected_indices = np.random.choice(self.n_features, n_selected, replace=False)
        for idx in selected_indices:
            genes[idx] = True
        return genes
    
    def calculate_fitness(self) -> float:
        # Use a simple classifier with selected features
        selected_features = [i for i, selected in enumerate(self.genes) if selected]
        
        if not selected_features:
            return 0.0  # No features selected
        
        # Simple scoring: could be replaced with actual ML model
        # This is just a placeholder
        n_selected = len(selected_features)
        feature_penalty = n_selected / self.n_features  # Prefer fewer features
        
        # Simulate accuracy (replace with real model evaluation)
        simulated_accuracy = 0.8 + 0.1 * np.random.random()
        
        return simulated_accuracy - 0.1 * feature_penalty

# Usage
problem_data = {
    'X_train': np.random.randn(100, 20),
    'y_train': np.random.randint(0, 2, 100),
    'X_val': np.random.randn(30, 20),
    'y_val': np.random.randint(0, 2, 30),
    'n_features': 20
}

ga = GeneticAlgorithm(
    individual_class=FeatureSelectionIndividual,
    population_size=50,
    generations=100,
    problem_data=problem_data
)

best_individual, stats_history = ga.evolve()
selected_features = [i for i, selected in enumerate(best_individual.genes) if selected]
print(f"Selected features: {selected_features}")
```

## Best Practices

### 1. Problem Representation
- Choose the right gene representation for your problem
- Ensure your fitness function properly ranks solutions
- Handle constraints appropriately (repair, penalty, or rejection)

### 2. Parameter Tuning
- Start with standard parameters (pop_size=100, crossover_rate=0.8, mutation_rate=0.1)
- Use parameter sensitivity analysis to find optimal settings
- Consider adaptive parameters for complex problems

### 3. Termination Criteria
- Set reasonable termination criteria (generations, target fitness, stagnation)
- Monitor convergence to avoid unnecessary computation
- Use early stopping for time-critical applications

### 4. Debugging and Testing
- Test your fitness function thoroughly
- Verify that mutations produce valid solutions
- Check that crossover preserves problem constraints
- Use visualization tools to understand algorithm behavior

### 5. Performance Optimization
- Profile your fitness function - it's usually the bottleneck
- Consider parallel fitness evaluation for expensive functions
- Use elitism to preserve good solutions
- Implement problem-specific local search for hybrid algorithms

## Common Pitfalls

1. **Poor fitness function design**: Ensure it distinguishes between good and bad solutions
2. **Inappropriate representation**: Gene representation should match problem structure
3. **Parameter mismatching**: Population size too small, mutation rate too high/low
4. **Constraint violations**: Handle infeasible solutions properly
5. **Premature convergence**: Use diversity preservation techniques
6. **Inefficient implementation**: Optimize the fitness function and operators

This completes the implementation guide. For more advanced techniques, see the [Advanced Topics](05_advanced.md) documentation.