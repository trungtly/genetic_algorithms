# Example Problems and Solutions

This document provides detailed walkthroughs of the example problems included in this genetic algorithms framework. Each example demonstrates different aspects of genetic algorithm design and implementation.

## Table of Contents

1. [Traveling Salesman Problem (TSP)](#traveling-salesman-problem-tsp)
2. [Knapsack Problem](#knapsack-problem)
3. [Function Optimization Examples](#function-optimization-examples)
4. [Custom Problem Examples](#custom-problem-examples)
5. [Visualization Examples](#visualization-examples)

## Traveling Salesman Problem (TSP)

The TSP is a classic combinatorial optimization problem where we need to find the shortest route visiting all cities exactly once and returning to the starting city.

### Problem Setup

```python
from genetic_algorithms.examples.tsp import TSPProblem, TSPIndividual
from genetic_algorithms.core import GeneticAlgorithm

# Create a random TSP problem with 15 cities
problem = TSPProblem.create_random_problem(n_cities=15, seed=42)

print(f"Number of cities: {problem.n_cities}")
print(f"Distance matrix shape: {problem.distance_matrix.shape}")
print(f"City coordinates: {problem.coordinates[:3]}...")  # First 3 cities
```

### Algorithm Configuration

```python
# Set up the genetic algorithm for TSP
ga = GeneticAlgorithm(
    individual_class=TSPIndividual,
    population_size=100,
    generations=200,
    crossover_rate=0.8,
    mutation_rate=0.2,
    selection_method='tournament',
    tournament_size=5,
    elitism=True,
    elite_size=5,
    problem_data={'distance_matrix': problem.distance_matrix}
)
```

### Running the Algorithm

```python
# Run the algorithm
print("Starting TSP optimization...")
best_individual, stats_history = ga.evolve()

print(f"Best route distance: {best_individual.fitness:.2f}")
print(f"Best route: {best_individual.genes}")

# Calculate improvement
initial_fitness = stats_history[0]['best_fitness']
final_fitness = stats_history[-1]['best_fitness']
improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100
print(f"Improvement: {improvement:.1f}%")
```

### Understanding TSP Results

```python
# Analyze the solution
def analyze_tsp_solution(problem, best_individual, stats_history):
    route = best_individual.genes
    total_distance = best_individual.fitness
    
    print("\n=== TSP Solution Analysis ===")
    print(f"Route: {' -> '.join(map(str, route))} -> {route[0]}")
    print(f"Total distance: {total_distance:.2f}")
    
    # Calculate individual segment distances
    print("\nSegment distances:")
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        distance = problem.distance_matrix[from_city][to_city]
        print(f"  {from_city} -> {to_city}: {distance:.2f}")
    
    # Evolution statistics
    print(f"\nEvolution statistics:")
    print(f"  Generations: {len(stats_history)}")
    print(f"  Initial best fitness: {stats_history[0]['best_fitness']:.2f}")
    print(f"  Final best fitness: {stats_history[-1]['best_fitness']:.2f}")
    
    # Convergence analysis
    no_improvement_count = 0
    last_best = stats_history[0]['best_fitness']
    for stats in stats_history[1:]:
        if abs(stats['best_fitness'] - last_best) < 0.01:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        last_best = stats['best_fitness']
    
    print(f"  Generations without improvement at end: {no_improvement_count}")

analyze_tsp_solution(problem, best_individual, stats_history)
```

### Comparing with Optimal Solutions

```python
# For small problems, compare with brute force solution
if problem.n_cities <= 10:
    print("\nComparing with brute force solution...")
    optimal_route, optimal_distance = problem.solve_brute_force()
    
    print(f"Optimal route: {optimal_route}")
    print(f"Optimal distance: {optimal_distance:.2f}")
    print(f"GA distance: {best_individual.fitness:.2f}")
    
    gap = ((best_individual.fitness - optimal_distance) / optimal_distance) * 100
    print(f"Optimality gap: {gap:.2f}%")
else:
    print(f"\nProblem too large for brute force (n_cities = {problem.n_cities})")
```

### TSP Variants

```python
# Different problem types
problems = {
    'Random': TSPProblem.create_random_problem(10, seed=42),
    'Circle': TSPProblem.create_circle_problem(10, radius=50),
    'Grid': TSPProblem.create_grid_problem(3, 3, spacing=10),
    'Clustered': TSPProblem.create_clustered_problem(12, n_clusters=3, seed=42)
}

for name, prob in problems.items():
    ga = GeneticAlgorithm(
        individual_class=TSPIndividual,
        population_size=50,
        generations=100,
        problem_data={'distance_matrix': prob.distance_matrix}
    )
    
    best, stats = ga.evolve()
    print(f"{name} TSP - Best distance: {best.fitness:.2f}")
```

## Knapsack Problem

The 0/1 Knapsack Problem involves selecting items to maximize value while staying within weight capacity.

### Problem Setup

```python
from genetic_algorithms.examples.knapsack import KnapsackProblem, KnapsackIndividual

# Create a random knapsack problem
problem = KnapsackProblem.create_random_problem(
    n_items=20, 
    capacity=50, 
    seed=42
)

print(f"Number of items: {problem.n_items}")
print(f"Knapsack capacity: {problem.capacity}")
print("\nFirst 5 items:")
for i, item in enumerate(problem.items[:5]):
    print(f"  Item {i}: weight={item['weight']:.1f}, value={item['value']:.1f}, "
          f"ratio={item['value']/item['weight']:.2f}")
```

### Algorithm Configuration

```python
# Configure GA for knapsack problem
ga = GeneticAlgorithm(
    individual_class=KnapsackIndividual,
    population_size=80,
    generations=150,
    crossover_rate=0.7,
    mutation_rate=0.1,
    selection_method='tournament',
    elitism=True,
    problem_data={
        'items': problem.items,
        'capacity': problem.capacity
    }
)
```

### Running and Analyzing Results

```python
# Run the algorithm
print("Starting knapsack optimization...")
best_individual, stats_history = ga.evolve()

def analyze_knapsack_solution(problem, best_individual):
    solution = [bool(gene) for gene in best_individual.genes]
    selected_items = [problem.items[i] for i, selected in enumerate(solution) if selected]
    
    total_weight = sum(item['weight'] for item in selected_items)
    total_value = sum(item['value'] for item in selected_items)
    
    print("\n=== Knapsack Solution Analysis ===")
    print(f"Total value: {total_value:.1f}")
    print(f"Total weight: {total_weight:.1f} / {problem.capacity}")
    print(f"Capacity utilization: {100 * total_weight / problem.capacity:.1f}%")
    print(f"Number of items selected: {len(selected_items)} / {problem.n_items}")
    
    if selected_items:
        avg_ratio = sum(item['value'] / item['weight'] for item in selected_items) / len(selected_items)
        print(f"Average value/weight ratio of selected items: {avg_ratio:.2f}")
        
        print("\nSelected items:")
        for i, selected in enumerate(solution):
            if selected:
                item = problem.items[i]
                print(f"  Item {i}: weight={item['weight']:.1f}, "
                      f"value={item['value']:.1f}, ratio={item['value']/item['weight']:.2f}")
    
    return total_value, total_weight

total_value, total_weight = analyze_knapsack_solution(problem, best_individual)
```

### Comparing with Greedy Solution

```python
# Compare with greedy heuristic
greedy_value, greedy_weight, greedy_solution = problem.greedy_solution()

print(f"\n=== Comparison with Greedy Solution ===")
print(f"GA Solution - Value: {total_value:.1f}, Weight: {total_weight:.1f}")
print(f"Greedy Solution - Value: {greedy_value:.1f}, Weight: {greedy_weight:.1f}")

improvement = ((total_value - greedy_value) / greedy_value) * 100
print(f"GA improvement over greedy: {improvement:.1f}%")
```

### Knapsack Variants and Benchmarks

```python
# Standard benchmark problems
benchmark_problems = [
    {'n_items': 10, 'capacity': 25, 'name': 'Small'},
    {'n_items': 50, 'capacity': 100, 'name': 'Medium'},
    {'n_items': 100, 'capacity': 200, 'name': 'Large'}
]

for params in benchmark_problems:
    problem = KnapsackProblem.create_random_problem(
        params['n_items'], 
        params['capacity'], 
        seed=42
    )
    
    ga = GeneticAlgorithm(
        individual_class=KnapsackIndividual,
        population_size=100,
        generations=200,
        problem_data={'items': problem.items, 'capacity': problem.capacity}
    )
    
    best, stats = ga.evolve()
    greedy_value, _, _ = problem.greedy_solution()
    
    improvement = ((best.fitness - greedy_value) / greedy_value) * 100
    print(f"{params['name']} problem ({params['n_items']} items): "
          f"GA={best.fitness:.1f}, Greedy={greedy_value:.1f}, "
          f"Improvement={improvement:.1f}%")
```

## Function Optimization Examples

### Single-Objective Function Optimization

```python
import numpy as np
from genetic_algorithms.core import GeneticAlgorithm, RealValuedIndividual

# Define test functions
def sphere_function(x):
    """Simple sphere function: f(x) = -sum(x_i^2)"""
    return -sum(xi**2 for xi in x)

def rosenbrock_function(x):
    """Rosenbrock function - challenging optimization problem"""
    return -sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                for i in range(len(x) - 1))

def rastrigin_function(x):
    """Rastrigin function - multimodal optimization"""
    A = 10
    n = len(x)
    return -(A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x))

# Test functions
functions = {
    'Sphere': {'func': sphere_function, 'bounds': (-5, 5), 'optimum': 0, 'dims': 5},
    'Rosenbrock': {'func': rosenbrock_function, 'bounds': (-2, 2), 'optimum': 0, 'dims': 4},
    'Rastrigin': {'func': rastrigin_function, 'bounds': (-5.12, 5.12), 'optimum': 0, 'dims': 5}
}

for name, config in functions.items():
    print(f"\n=== Optimizing {name} Function ===")
    
    ga = GeneticAlgorithm(
        individual_class=RealValuedIndividual,
        population_size=100,
        generations=300,
        chromosome_length=config['dims'],
        gene_bounds=config['bounds'],
        fitness_function=config['func'],
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    best, stats = ga.evolve()
    
    print(f"Best solution: {[f'{x:.4f}' for x in best.genes]}")
    print(f"Best fitness: {best.fitness:.6f}")
    print(f"Known optimum: {config['optimum']}")
    print(f"Error: {abs(best.fitness - config['optimum']):.6f}")
```

### Constrained Optimization Example

```python
def constrained_optimization_example():
    """Example with constraints: maximize x1 + x2 subject to x1^2 + x2^2 <= 1"""
    
    def objective_function(x):
        """Objective: maximize x1 + x2"""
        return x[0] + x[1]
    
    def constraint_penalty(x):
        """Penalty for violating x1^2 + x2^2 <= 1"""
        violation = max(0, x[0]**2 + x[1]**2 - 1)
        return -1000 * violation  # Large penalty for constraint violation
    
    def fitness_function(individual):
        x = individual.genes
        return objective_function(x) + constraint_penalty(x)
    
    ga = GeneticAlgorithm(
        individual_class=RealValuedIndividual,
        population_size=100,
        generations=200,
        chromosome_length=2,
        gene_bounds=(-2, 2),
        fitness_function=fitness_function
    )
    
    best, stats = ga.evolve()
    
    print("\n=== Constrained Optimization Results ===")
    print(f"Best solution: x1={best.genes[0]:.4f}, x2={best.genes[1]:.4f}")
    print(f"Objective value: {sum(best.genes):.4f}")
    print(f"Constraint value: {sum(x**2 for x in best.genes):.4f} (should be ≤ 1)")
    print(f"Constraint satisfied: {sum(x**2 for x in best.genes) <= 1.001}")  # Small tolerance
    
    # Theoretical optimum: x1=x2=1/sqrt(2) ≈ 0.707, objective = sqrt(2) ≈ 1.414
    theoretical_optimum = 2 / np.sqrt(2)
    print(f"Theoretical optimum: {theoretical_optimum:.4f}")

constrained_optimization_example()
```

## Custom Problem Examples

### Feature Selection for Machine Learning

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from genetic_algorithms.core.individual import Individual

class FeatureSelectionIndividual(Individual):
    """Individual for feature selection using GA."""
    
    def __init__(self, genes=None, **kwargs):
        self.X_train = kwargs.get('problem_data', {}).get('X_train')
        self.y_train = kwargs.get('problem_data', {}).get('y_train')
        self.X_val = kwargs.get('problem_data', {}).get('X_val')
        self.y_val = kwargs.get('problem_data', {}).get('y_val')
        self.n_features = self.X_train.shape[1] if self.X_train is not None else 20
        
        if genes is None:
            genes = self.generate_random_genes()
        
        super().__init__(genes)
    
    def generate_random_genes(self):
        # Ensure at least one feature is selected
        genes = [False] * self.n_features
        n_selected = np.random.randint(1, max(2, self.n_features // 3))
        selected_indices = np.random.choice(self.n_features, n_selected, replace=False)
        for idx in selected_indices:
            genes[idx] = True
        return genes
    
    def calculate_fitness(self):
        selected_features = [i for i, selected in enumerate(self.genes) if selected]
        
        if not selected_features:
            return 0.0
        
        # Train model with selected features
        X_train_selected = self.X_train[:, selected_features]
        X_val_selected = self.X_val[:, selected_features]
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_selected, self.y_train)
        
        predictions = model.predict(X_val_selected)
        accuracy = accuracy_score(self.y_val, predictions)
        
        # Balance accuracy and number of features
        feature_penalty = len(selected_features) / self.n_features
        return accuracy - 0.1 * feature_penalty

def feature_selection_example():
    """Complete feature selection example."""
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    problem_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }
    
    # Run feature selection
    ga = GeneticAlgorithm(
        individual_class=FeatureSelectionIndividual,
        population_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        problem_data=problem_data
    )
    
    print("\n=== Feature Selection Example ===")
    print("Running genetic algorithm for feature selection...")
    
    best, stats = ga.evolve()
    
    selected_features = [i for i, selected in enumerate(best.genes) if selected]
    
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Feature indices: {selected_features}")
    print(f"Best fitness (accuracy - penalty): {best.fitness:.4f}")
    
    # Compare with using all features
    model_all = RandomForestClassifier(n_estimators=50, random_state=42)
    model_all.fit(X_train, y_train)
    accuracy_all = accuracy_score(y_val, model_all.predict(X_val))
    
    print(f"Accuracy with all features: {accuracy_all:.4f}")
    
    return selected_features

selected_features = feature_selection_example()
```

### Neural Network Architecture Search

```python
class NeuralArchitectureIndividual(Individual):
    """Individual representing neural network architecture."""
    
    def __init__(self, genes=None, **kwargs):
        self.max_layers = kwargs.get('problem_data', {}).get('max_layers', 5)
        self.max_neurons = kwargs.get('problem_data', {}).get('max_neurons', 128)
        
        if genes is None:
            genes = self.generate_random_genes()
        
        super().__init__(genes)
    
    def generate_random_genes(self):
        # Gene format: [n_layers, layer1_neurons, layer2_neurons, ...]
        n_layers = np.random.randint(1, self.max_layers + 1)
        genes = [n_layers]
        
        for _ in range(n_layers):
            neurons = np.random.randint(16, self.max_neurons + 1)
            genes.append(neurons)
        
        # Pad with zeros for unused layers
        while len(genes) < self.max_layers + 1:
            genes.append(0)
        
        return genes
    
    def calculate_fitness(self):
        n_layers = self.genes[0]
        layer_sizes = [self.genes[i+1] for i in range(n_layers) if self.genes[i+1] > 0]
        
        if not layer_sizes:
            return 0.0
        
        # Simulate network training (replace with actual training)
        # Fitness based on accuracy and network complexity
        
        # Simulated accuracy (would be real validation accuracy)
        base_accuracy = 0.85
        layer_bonus = min(0.05, n_layers * 0.01)  # Slight bonus for depth
        complexity_penalty = sum(layer_sizes) / (self.max_neurons * self.max_layers)
        
        simulated_accuracy = base_accuracy + layer_bonus - 0.1 * complexity_penalty
        simulated_accuracy += np.random.normal(0, 0.02)  # Add noise
        
        return max(0, simulated_accuracy)

def neural_architecture_search_example():
    """Example of neural architecture search using GA."""
    
    ga = GeneticAlgorithm(
        individual_class=NeuralArchitectureIndividual,
        population_size=20,
        generations=30,
        problem_data={'max_layers': 4, 'max_neurons': 64}
    )
    
    print("\n=== Neural Architecture Search Example ===")
    best, stats = ga.evolve()
    
    n_layers = best.genes[0]
    architecture = [best.genes[i+1] for i in range(n_layers) if best.genes[i+1] > 0]
    
    print(f"Best architecture: {architecture}")
    print(f"Number of layers: {n_layers}")
    print(f"Total parameters (approx): {sum(architecture) * 2}")  # Simplified calculation
    print(f"Best fitness: {best.fitness:.4f}")

neural_architecture_search_example()
```

## Visualization Examples

### Complete Visualization Workflow

```python
from genetic_algorithms.visualization import EvolutionPlotter, SolutionVisualizer, DiversityAnalyzer

def complete_visualization_example():
    """Demonstrate all visualization capabilities."""
    
    # 1. TSP with visualization
    print("=== TSP with Visualization ===")
    problem = TSPProblem.create_random_problem(12, seed=42)
    
    ga = GeneticAlgorithm(
        individual_class=TSPIndividual,
        population_size=50,
        generations=100,
        problem_data={'distance_matrix': problem.distance_matrix}
    )
    
    best, stats = ga.evolve()
    
    # Create visualizers
    evolution_plotter = EvolutionPlotter()
    solution_visualizer = SolutionVisualizer()
    diversity_analyzer = DiversityAnalyzer()
    
    # Plot evolution
    evolution_plotter.plot_fitness_evolution(stats, title="TSP Evolution")
    
    # Visualize best route
    solution_visualizer.visualize_tsp_route(
        problem.coordinates, 
        best.genes,
        title="Best TSP Route"
    )
    
    # Convergence analysis
    evolution_plotter.plot_convergence_analysis(stats, title="TSP Convergence")
    
    # 2. Parameter comparison
    print("\n=== Parameter Comparison ===")
    mutation_rates = [0.05, 0.1, 0.2, 0.3]
    comparison_data = {}
    
    for rate in mutation_rates:
        ga_test = GeneticAlgorithm(
            individual_class=TSPIndividual,
            population_size=30,
            generations=50,
            mutation_rate=rate,
            problem_data={'distance_matrix': problem.distance_matrix}
        )
        
        _, test_stats = ga_test.evolve()
        comparison_data[f"Mutation Rate {rate}"] = test_stats
    
    evolution_plotter.compare_multiple_runs(
        comparison_data,
        title="Mutation Rate Comparison"
    )
    
    # 3. Function optimization landscape
    print("\n=== Function Landscape Visualization ===")
    
    def simple_2d_function(point):
        x, y = point
        return -(x**2 + y**2)  # Simple paraboloid
    
    # Generate sample population
    population_2d = [(np.random.uniform(-3, 3), np.random.uniform(-3, 3)) 
                     for _ in range(20)]
    
    solution_visualizer.visualize_function_landscape(
        simple_2d_function,
        bounds=((-3, 3), (-3, 3)),
        best_solution=(0, 0),
        population=population_2d,
        title="Function Optimization Landscape"
    )

# Run the complete visualization example
complete_visualization_example()
```

### Creating Custom Visualizations

```python
def custom_visualization_example():
    """Show how to create custom visualizations."""
    
    import matplotlib.pyplot as plt
    
    # Run a simple GA
    problem = KnapsackProblem.create_random_problem(15, capacity=30, seed=42)
    
    ga = GeneticAlgorithm(
        individual_class=KnapsackIndividual,
        population_size=40,
        generations=80,
        problem_data={'items': problem.items, 'capacity': problem.capacity}
    )
    
    best, stats = ga.evolve()
    
    # Custom plot: Fitness vs Generation with custom styling
    generations = range(len(stats))
    best_fitness = [s['best_fitness'] for s in stats]
    avg_fitness = [s['avg_fitness'] for s in stats]
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Fitness Evolution
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_fitness, 'g-', linewidth=3, label='Best', alpha=0.8)
    plt.plot(generations, avg_fitness, 'b--', linewidth=2, label='Average', alpha=0.7)
    plt.fill_between(generations, best_fitness, avg_fitness, alpha=0.2, color='green')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Value)')
    plt.title('Knapsack Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Solution Quality Distribution
    plt.subplot(1, 2, 2)
    final_population_fitness = [individual.fitness for individual in ga.population.individuals]
    
    plt.hist(final_population_fitness, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(best.fitness, color='red', linestyle='--', linewidth=2, 
                label=f'Best: {best.fitness:.1f}')
    plt.axvline(np.mean(final_population_fitness), color='orange', linestyle='-', 
                linewidth=2, label=f'Mean: {np.mean(final_population_fitness):.1f}')
    
    plt.xlabel('Fitness Value')
    plt.ylabel('Frequency')
    plt.title('Final Population Fitness Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Best solution value: {best.fitness:.1f}")
    print(f"Population fitness std: {np.std(final_population_fitness):.2f}")

custom_visualization_example()
```

## Performance Benchmarking

### Systematic Performance Testing

```python
def benchmark_performance():
    """Comprehensive performance benchmarking."""
    
    print("=== Performance Benchmarking ===")
    
    # Test different problem sizes
    tsp_sizes = [8, 10, 15, 20]
    results = {}
    
    for size in tsp_sizes:
        print(f"\nTesting TSP with {size} cities...")
        
        problem = TSPProblem.create_random_problem(size, seed=42)
        
        import time
        start_time = time.time()
        
        ga = GeneticAlgorithm(
            individual_class=TSPIndividual,
            population_size=min(100, size * 8),  # Scale population with problem size
            generations=min(200, size * 15),      # Scale generations
            problem_data={'distance_matrix': problem.distance_matrix}
        )
        
        best, stats = ga.evolve()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        results[size] = {
            'runtime': runtime,
            'best_fitness': best.fitness,
            'generations': len(stats),
            'final_diversity': stats[-1].get('diversity', 0)
        }
        
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Best distance: {best.fitness:.2f}")
        print(f"  Generations: {len(stats)}")
    
    # Summary
    print("\n=== Benchmark Summary ===")
    print("Size\tRuntime(s)\tBest Distance\tGenerations")
    for size, data in results.items():
        print(f"{size}\t{data['runtime']:.2f}\t\t{data['best_fitness']:.2f}\t\t{data['generations']}")

# Run benchmarking
benchmark_performance()
```

This completes the comprehensive examples documentation, showing practical usage of all major components in the genetic algorithms framework.