"""
Population diversity analysis and visualization tools.

This module provides tools to analyze and visualize population diversity:
- Genetic diversity metrics
- Phenotypic diversity analysis
- Diversity vs fitness trade-offs
- Population clustering visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import Counter
import itertools


class DiversityAnalyzer:
    """Analyzes and visualizes population diversity."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the diversity analyzer.
        
        Args:
            style: Matplotlib style to use
            figsize: Figure size as (width, height)
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
    
    def calculate_hamming_diversity(self, population: List[List[int]]) -> float:
        """
        Calculate average Hamming distance diversity for binary populations.
        
        Args:
            population: List of binary individuals
            
        Returns:
            Average Hamming distance between all pairs
        """
        if len(population) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def calculate_euclidean_diversity(self, population: List[List[float]]) -> float:
        """
        Calculate average Euclidean distance diversity for real-valued populations.
        
        Args:
            population: List of real-valued individuals
            
        Returns:
            Average Euclidean distance between all pairs
        """
        if len(population) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.sqrt(sum((a - b)**2 for a, b in zip(population[i], population[j])))
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def calculate_fitness_diversity(self, fitness_values: List[float]) -> Dict[str, float]:
        """
        Calculate fitness-based diversity metrics.
        
        Args:
            fitness_values: List of fitness values
            
        Returns:
            Dictionary with diversity metrics
        """
        if not fitness_values:
            return {'variance': 0.0, 'std_dev': 0.0, 'range': 0.0, 'coefficient_of_variation': 0.0}
        
        fitness_array = np.array(fitness_values)
        variance = np.var(fitness_array)
        std_dev = np.std(fitness_array)
        fitness_range = np.max(fitness_array) - np.min(fitness_array)
        mean_fitness = np.mean(fitness_array)
        cv = std_dev / mean_fitness if mean_fitness != 0 else 0.0
        
        return {
            'variance': variance,
            'std_dev': std_dev,
            'range': fitness_range,
            'coefficient_of_variation': cv
        }
    
    def plot_diversity_evolution(self, 
                               diversity_history: List[float],
                               fitness_history: List[float],
                               title: str = "Diversity vs Fitness Evolution",
                               save_path: Optional[str] = None) -> None:
        """
        Plot diversity and fitness evolution over generations.
        
        Args:
            diversity_history: List of diversity values per generation
            fitness_history: List of best fitness values per generation
            title: Plot title
            save_path: Path to save the plot
        """
        if len(diversity_history) != len(fitness_history):
            raise ValueError("Diversity and fitness histories must have same length")
        
        generations = range(len(diversity_history))
        
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Plot diversity
        color = 'tab:blue'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Population Diversity', color=color)
        ax1.plot(generations, diversity_history, color=color, linewidth=2, alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Plot fitness on second y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Best Fitness', color=color)
        ax2.plot(generations, fitness_history, color=color, linewidth=2, alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_population_distribution(self, 
                                   population_data: List[List[float]],
                                   fitness_values: List[float],
                                   title: str = "Population Distribution",
                                   max_dimensions: int = 3,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize population distribution in solution space.
        
        Args:
            population_data: List of individuals (each as list of values)
            fitness_values: Fitness values for coloring
            title: Plot title
            max_dimensions: Maximum dimensions to plot (1, 2, or 3)
            save_path: Path to save the plot
        """
        if not population_data:
            print("No population data to visualize")
            return
        
        dimensions = len(population_data[0])
        plot_dims = min(dimensions, max_dimensions)
        
        if plot_dims == 1:
            plt.figure(figsize=self.figsize)
            values = [ind[0] for ind in population_data]
            plt.scatter(values, fitness_values, c=fitness_values, cmap='viridis', alpha=0.7)
            plt.xlabel('Dimension 1')
            plt.ylabel('Fitness')
            plt.colorbar(label='Fitness')
            
        elif plot_dims == 2:
            plt.figure(figsize=self.figsize)
            x_vals = [ind[0] for ind in population_data]
            y_vals = [ind[1] for ind in population_data]
            scatter = plt.scatter(x_vals, y_vals, c=fitness_values, cmap='viridis', alpha=0.7)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.colorbar(scatter, label='Fitness')
            
        elif plot_dims >= 3:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            x_vals = [ind[0] for ind in population_data]
            y_vals = [ind[1] for ind in population_data]
            z_vals = [ind[2] for ind in population_data]
            scatter = ax.scatter(x_vals, y_vals, z_vals, c=fitness_values, cmap='viridis', alpha=0.7)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            plt.colorbar(scatter, label='Fitness')
        
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_selection_pressure(self, 
                                 fitness_history: List[List[float]],
                                 title: str = "Selection Pressure Analysis",
                                 save_path: Optional[str] = None) -> None:
        """
        Analyze selection pressure over generations.
        
        Args:
            fitness_history: List of fitness arrays (one per generation)
            title: Plot title
            save_path: Path to save the plot
        """
        generations = range(len(fitness_history))
        
        # Calculate metrics for each generation
        selection_intensity = []  # (best - avg) / std
        takeover_time = []  # Generations for best to dominate
        
        for gen_fitness in fitness_history:
            if len(gen_fitness) == 0:
                continue
                
            best_fit = max(gen_fitness)
            avg_fit = np.mean(gen_fitness)
            std_fit = np.std(gen_fitness)
            
            # Selection intensity
            intensity = (best_fit - avg_fit) / std_fit if std_fit > 0 else 0
            selection_intensity.append(intensity)
        
        # Plot selection pressure metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Selection intensity over time
        ax1.plot(generations[:len(selection_intensity)], selection_intensity, 
                'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Selection Intensity')
        ax1.set_title('Selection Intensity Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Fitness distribution changes
        if len(fitness_history) > 1:
            # Show fitness distributions for first, middle, and last generations
            generations_to_plot = [0, len(fitness_history)//2, -1]
            colors = ['red', 'orange', 'green']
            labels = ['Early', 'Middle', 'Late']
            
            for i, (gen_idx, color, label) in enumerate(zip(generations_to_plot, colors, labels)):
                gen_fitness = fitness_history[gen_idx]
                ax2.hist(gen_fitness, bins=20, alpha=0.5, color=color, label=f'{label} (Gen {gen_idx if gen_idx >= 0 else len(fitness_history)-1})')
            
            ax2.set_xlabel('Fitness')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Fitness Distribution Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_diversity_vs_performance(self, 
                                    diversity_values: List[float],
                                    performance_values: List[float],
                                    title: str = "Diversity vs Performance",
                                    save_path: Optional[str] = None) -> None:
        """
        Plot relationship between diversity and performance.
        
        Args:
            diversity_values: List of diversity measurements
            performance_values: List of corresponding performance values
            title: Plot title
            save_path: Path to save the plot
        """
        if len(diversity_values) != len(performance_values):
            raise ValueError("Diversity and performance lists must have same length")
        
        plt.figure(figsize=self.figsize)
        
        # Scatter plot
        plt.scatter(diversity_values, performance_values, alpha=0.6, s=50)
        
        # Add trend line
        if len(diversity_values) > 1:
            z = np.polyfit(diversity_values, performance_values, 1)
            p = np.poly1d(z)
            plt.plot(diversity_values, p(diversity_values), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            correlation = np.corrcoef(diversity_values, performance_values)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('Population Diversity')
        plt.ylabel('Performance (Best Fitness)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()