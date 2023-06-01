"""
Evolution plotting utilities for genetic algorithms.

This module provides tools to visualize the evolution process including:
- Fitness progression over generations
- Best/average/worst fitness trends
- Convergence analysis
- Multi-run comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class EvolutionPlotter:
    """Plots evolution statistics and fitness progression."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the evolution plotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Figure size as (width, height)
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
    
    def plot_fitness_evolution(self, 
                             stats_history: List[Dict[str, Any]], 
                             title: str = "Fitness Evolution",
                             save_path: Optional[str] = None,
                             show_diversity: bool = False) -> None:
        """
        Plot fitness evolution over generations.
        
        Args:
            stats_history: List of statistics dictionaries from GA runs
            title: Plot title
            save_path: Path to save the plot (optional)
            show_diversity: Whether to show diversity metrics
        """
        if not stats_history:
            raise ValueError("No statistics history provided")
        
        generations = range(len(stats_history))
        best_fitness = [stats['best_fitness'] for stats in stats_history]
        avg_fitness = [stats['avg_fitness'] for stats in stats_history]
        worst_fitness = [stats['worst_fitness'] for stats in stats_history]
        
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Plot fitness curves
        ax1.plot(generations, best_fitness, 'g-', linewidth=2, label='Best', alpha=0.8)
        ax1.plot(generations, avg_fitness, 'b--', linewidth=1.5, label='Average', alpha=0.7)
        ax1.plot(generations, worst_fitness, 'r:', linewidth=1, label='Worst', alpha=0.6)
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add diversity plot if requested
        if show_diversity and 'diversity' in stats_history[0]:
            ax2 = ax1.twinx()
            diversity = [stats['diversity'] for stats in stats_history]
            ax2.plot(generations, diversity, 'm-', linewidth=1.5, alpha=0.6, label='Diversity')
            ax2.set_ylabel('Population Diversity', color='m')
            ax2.tick_params(axis='y', labelcolor='m')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_convergence_analysis(self, 
                                stats_history: List[Dict[str, Any]],
                                convergence_threshold: float = 0.01,
                                window_size: int = 10,
                                title: str = "Convergence Analysis",
                                save_path: Optional[str] = None) -> None:
        """
        Analyze and plot convergence behavior.
        
        Args:
            stats_history: List of statistics dictionaries
            convergence_threshold: Threshold for considering convergence
            window_size: Window size for convergence detection
            title: Plot title
            save_path: Path to save the plot
        """
        if len(stats_history) < window_size:
            raise ValueError(f"Need at least {window_size} generations for convergence analysis")
        
        generations = range(len(stats_history))
        best_fitness = [stats['best_fitness'] for stats in stats_history]
        
        # Calculate fitness improvement rate
        improvement_rate = []
        for i in range(window_size, len(best_fitness)):
            window_start = best_fitness[i - window_size]
            window_end = best_fitness[i]
            rate = abs(window_end - window_start) / window_size if window_start != 0 else 0
            improvement_rate.append(rate)
        
        # Detect convergence point
        convergence_gen = None
        for i, rate in enumerate(improvement_rate):
            if rate < convergence_threshold:
                convergence_gen = i + window_size
                break
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Plot fitness evolution
        ax1.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness')
        if convergence_gen:
            ax1.axvline(x=convergence_gen, color='r', linestyle='--', alpha=0.7, 
                       label=f'Convergence (Gen {convergence_gen})')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title(f'{title} - Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot improvement rate
        improvement_gens = range(window_size, len(best_fitness))
        ax2.plot(improvement_gens, improvement_rate, 'b-', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=convergence_threshold, color='r', linestyle=':', alpha=0.7,
                   label=f'Threshold ({convergence_threshold})')
        if convergence_gen:
            ax2.axvline(x=convergence_gen, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Improvement Rate')
        ax2.set_title('Fitness Improvement Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_multiple_runs(self, 
                            runs_data: Dict[str, List[Dict[str, Any]]],
                            metric: str = 'best_fitness',
                            title: str = "Multi-Run Comparison",
                            save_path: Optional[str] = None) -> None:
        """
        Compare multiple GA runs or parameter settings.
        
        Args:
            runs_data: Dictionary mapping run names to statistics histories
            metric: Metric to compare ('best_fitness', 'avg_fitness', etc.)
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))
        
        for i, (run_name, stats_history) in enumerate(runs_data.items()):
            generations = range(len(stats_history))
            values = [stats[metric] for stats in stats_history]
            
            plt.plot(generations, values, color=colors[i], linewidth=2, 
                    label=run_name, alpha=0.8)
        
        plt.xlabel('Generation')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_parameter_sensitivity(self, 
                                 parameter_results: Dict[str, List[float]],
                                 parameter_name: str,
                                 title: str = "Parameter Sensitivity Analysis",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot sensitivity analysis for a parameter.
        
        Args:
            parameter_results: Dict mapping parameter values to final fitness results
            parameter_name: Name of the parameter being analyzed
            title: Plot title
            save_path: Path to save the plot
        """
        param_values = []
        mean_fitness = []
        std_fitness = []
        
        for param_val, fitness_results in parameter_results.items():
            param_values.append(param_val)
            mean_fitness.append(np.mean(fitness_results))
            std_fitness.append(np.std(fitness_results))
        
        # Sort by parameter values
        sorted_data = sorted(zip(param_values, mean_fitness, std_fitness))
        param_values, mean_fitness, std_fitness = zip(*sorted_data)
        
        plt.figure(figsize=self.figsize)
        plt.errorbar(param_values, mean_fitness, yerr=std_fitness, 
                    marker='o', linewidth=2, capsize=5, capthick=2)
        
        plt.xlabel(parameter_name)
        plt.ylabel('Final Best Fitness')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()