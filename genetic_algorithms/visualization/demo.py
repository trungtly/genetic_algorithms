"""
Demonstration of genetic algorithm visualization tools.

This script shows how to use the visualization tools with the TSP and Knapsack examples.
Run this script to see various visualization capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from genetic_algorithms.core import GeneticAlgorithm
from genetic_algorithms.examples.tsp import TSPProblem, TSPIndividual
from genetic_algorithms.examples.knapsack import KnapsackProblem, KnapsackIndividual
from genetic_algorithms.visualization import EvolutionPlotter, SolutionVisualizer, DiversityAnalyzer


def demo_tsp_visualization():
    """Demonstrate TSP visualization capabilities."""
    print("=== TSP Visualization Demo ===")
    
    # Create a TSP problem
    problem = TSPProblem.create_random_problem(15, seed=42)
    
    # Set up GA
    ga = GeneticAlgorithm(
        individual_class=TSPIndividual,
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.2,
        problem_data={'distance_matrix': problem.distance_matrix}
    )
    
    # Run GA
    print("Running TSP genetic algorithm...")
    best_individual, stats_history = ga.evolve()
    
    # Create visualizers
    evolution_plotter = EvolutionPlotter()
    solution_visualizer = SolutionVisualizer()
    diversity_analyzer = DiversityAnalyzer()
    
    # 1. Plot fitness evolution
    print("Plotting fitness evolution...")
    evolution_plotter.plot_fitness_evolution(
        stats_history, 
        title="TSP Fitness Evolution",
        show_diversity=True
    )
    
    # 2. Visualize best route
    print("Visualizing best TSP route...")
    solution_visualizer.visualize_tsp_route(
        problem.coordinates,
        best_individual.genes,
        title=f"Best TSP Route (Distance: {best_individual.fitness:.2f})"
    )
    
    # 3. Convergence analysis
    print("Analyzing convergence...")
    evolution_plotter.plot_convergence_analysis(
        stats_history,
        convergence_threshold=0.1,
        title="TSP Convergence Analysis"
    )
    
    # 4. Diversity analysis
    print("Analyzing population diversity...")
    diversity_history = [stats.get('diversity', 0) for stats in stats_history]
    fitness_history = [stats['best_fitness'] for stats in stats_history]
    
    diversity_analyzer.plot_diversity_evolution(
        diversity_history,
        fitness_history,
        title="TSP Diversity vs Fitness"
    )


def demo_knapsack_visualization():
    """Demonstrate Knapsack visualization capabilities."""
    print("\n=== Knapsack Visualization Demo ===")
    
    # Create a knapsack problem
    problem = KnapsackProblem.create_random_problem(20, capacity=50, seed=42)
    
    # Set up GA
    ga = GeneticAlgorithm(
        individual_class=KnapsackIndividual,
        population_size=50,
        generations=80,
        crossover_rate=0.7,
        mutation_rate=0.1,
        problem_data={'items': problem.items, 'capacity': problem.capacity}
    )
    
    # Run GA
    print("Running Knapsack genetic algorithm...")
    best_individual, stats_history = ga.evolve()
    
    # Create visualizers
    evolution_plotter = EvolutionPlotter()
    solution_visualizer = SolutionVisualizer()
    
    # 1. Plot fitness evolution
    print("Plotting fitness evolution...")
    evolution_plotter.plot_fitness_evolution(
        stats_history,
        title="Knapsack Fitness Evolution"
    )
    
    # 2. Visualize best solution
    print("Visualizing best knapsack solution...")
    solution_visualizer.visualize_knapsack_solution(
        problem.items,
        [bool(gene) for gene in best_individual.genes],
        problem.capacity,
        title=f"Best Knapsack Solution (Value: {best_individual.fitness:.1f})"
    )


def demo_parameter_sensitivity():
    """Demonstrate parameter sensitivity analysis."""
    print("\n=== Parameter Sensitivity Demo ===")
    
    # Test different mutation rates on TSP
    problem = TSPProblem.create_random_problem(10, seed=42)
    mutation_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    
    results = {}
    
    print("Testing different mutation rates...")
    for rate in mutation_rates:
        print(f"  Testing mutation rate: {rate}")
        
        # Run multiple trials
        trial_results = []
        for trial in range(3):  # Reduced for demo
            ga = GeneticAlgorithm(
                individual_class=TSPIndividual,
                population_size=30,
                generations=50,
                crossover_rate=0.8,
                mutation_rate=rate,
                problem_data={'distance_matrix': problem.distance_matrix}
            )
            
            best_individual, _ = ga.evolve()
            trial_results.append(best_individual.fitness)
        
        results[rate] = trial_results
    
    # Plot sensitivity analysis  
    evolution_plotter = EvolutionPlotter()
    evolution_plotter.plot_parameter_sensitivity(
        results,
        "Mutation Rate",
        title="TSP: Mutation Rate Sensitivity Analysis"
    )


def demo_multiple_runs_comparison():
    """Demonstrate comparison of multiple algorithm runs."""
    print("\n=== Multiple Runs Comparison Demo ===")
    
    problem = TSPProblem.create_random_problem(8, seed=42)
    
    # Compare different selection methods
    selection_methods = ['tournament', 'roulette']
    runs_data = {}
    
    print("Comparing selection methods...")
    for method in selection_methods:
        print(f"  Running with {method} selection...")
        
        ga = GeneticAlgorithm(
            individual_class=TSPIndividual,
            population_size=40,
            generations=60,
            crossover_rate=0.8,
            mutation_rate=0.2,
            selection_method=method,
            problem_data={'distance_matrix': problem.distance_matrix}
        )
        
        _, stats_history = ga.evolve()
        runs_data[f"{method.title()} Selection"] = stats_history
    
    # Plot comparison
    evolution_plotter = EvolutionPlotter()
    evolution_plotter.compare_multiple_runs(
        runs_data,
        metric='best_fitness',
        title="Selection Method Comparison"
    )


def demo_function_optimization():
    """Demonstrate function optimization visualization."""
    print("\n=== Function Optimization Demo ===")
    
    # Define a simple 2D function (Himmelblau's function)
    def himmelblau(point):
        x, y = point
        return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)  # Negative for maximization
    
    # Create some sample points
    best_solution = (3.0, 2.0)  # One of the global maxima
    population = [(np.random.uniform(-5, 5), np.random.uniform(-5, 5)) for _ in range(20)]
    
    # Visualize landscape
    solution_visualizer = SolutionVisualizer()
    solution_visualizer.visualize_function_landscape(
        himmelblau,
        bounds=((-5, 5), (-5, 5)),
        best_solution=best_solution,
        population=population,
        title="Himmelblau's Function Landscape"
    )


if __name__ == "__main__":
    print("Genetic Algorithm Visualization Demo")
    print("This demo will show various visualization capabilities.")
    print("Close each plot window to proceed to the next visualization.\n")
    
    try:
        # Run all demos
        demo_tsp_visualization()
        demo_knapsack_visualization()
        demo_parameter_sensitivity()
        demo_multiple_runs_comparison()
        demo_function_optimization()
        
        print("\n=== Demo Complete ===")
        print("All visualization demos completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()