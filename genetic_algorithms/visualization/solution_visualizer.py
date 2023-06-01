"""
Solution-specific visualization tools.

This module provides specialized visualizers for different types of problems:
- TSP route visualization
- Knapsack content visualization
- Function optimization landscapes
- Custom solution representations
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx


class SolutionVisualizer:
    """Visualizes problem-specific solutions."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the solution visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Figure size as (width, height)
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
    
    def visualize_tsp_route(self, 
                           coordinates: List[Tuple[float, float]], 
                           route: List[int],
                           title: str = "TSP Route",
                           save_path: Optional[str] = None,
                           show_labels: bool = True,
                           highlight_start: bool = True) -> None:
        """
        Visualize a TSP route on a 2D map.
        
        Args:
            coordinates: List of (x, y) coordinates for each city
            route: List of city indices representing the route
            title: Plot title
            save_path: Path to save the plot
            show_labels: Whether to show city labels
            highlight_start: Whether to highlight the starting city
        """
        if len(coordinates) != len(set(route)):
            raise ValueError("Number of coordinates must match route length")
        
        plt.figure(figsize=self.figsize)
        
        # Extract coordinates in route order
        x_coords = [coordinates[i][0] for i in route] + [coordinates[route[0]][0]]
        y_coords = [coordinates[i][1] for i in route] + [coordinates[route[0]][1]]
        
        # Plot cities
        for i, (x, y) in enumerate(coordinates):
            color = 'red' if highlight_start and i == route[0] else 'blue'
            size = 100 if highlight_start and i == route[0] else 50
            plt.scatter(x, y, c=color, s=size, alpha=0.7, zorder=3)
            
            if show_labels:
                plt.annotate(str(i), (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        # Plot route
        plt.plot(x_coords, y_coords, 'g-', linewidth=2, alpha=0.6, zorder=1)
        
        # Calculate and display route distance
        total_distance = 0
        for i in range(len(route)):
            city1 = route[i]
            city2 = route[(i + 1) % len(route)]
            dist = np.sqrt((coordinates[city1][0] - coordinates[city2][0])**2 + 
                          (coordinates[city1][1] - coordinates[city2][1])**2)
            total_distance += dist
        
        plt.title(f"{title}\nTotal Distance: {total_distance:.2f}")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_knapsack_solution(self, 
                                  items: List[Dict[str, Any]], 
                                  solution: List[bool],
                                  capacity: float,
                                  title: str = "Knapsack Solution",
                                  save_path: Optional[str] = None) -> None:
        """
        Visualize knapsack solution with items and capacity utilization.
        
        Args:
            items: List of item dictionaries with 'weight', 'value', and optionally 'name'
            solution: Binary list indicating which items are selected
            capacity: Knapsack capacity
            title: Plot title
            save_path: Path to save the plot
        """
        if len(items) != len(solution):
            raise ValueError("Number of items must match solution length")
        
        selected_items = [items[i] for i, selected in enumerate(solution) if selected]
        
        if not selected_items:
            print("No items selected in the solution")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Value vs Weight scatter plot
        weights = [item['weight'] for item in items]
        values = [item['value'] for item in items]
        colors = ['green' if selected else 'red' for selected in solution]
        
        ax1.scatter(weights, values, c=colors, alpha=0.6, s=50)
        ax1.set_xlabel('Weight')
        ax1.set_ylabel('Value')
        ax1.set_title('Items: Selected (Green) vs Not Selected (Red)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Capacity utilization
        total_weight = sum(item['weight'] for item in selected_items)
        total_value = sum(item['value'] for item in selected_items)
        
        ax2.bar(['Used', 'Remaining'], [total_weight, capacity - total_weight], 
                color=['orange', 'lightgray'], alpha=0.7)
        ax2.set_ylabel('Weight')
        ax2.set_title(f'Capacity Utilization\n({total_weight:.1f}/{capacity} = {100*total_weight/capacity:.1f}%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Value-to-weight ratio comparison
        ratios = [item['value'] / item['weight'] if item['weight'] > 0 else 0 for item in items]
        item_indices = range(len(items))
        
        bars = ax3.bar(item_indices, ratios, color=colors, alpha=0.6)
        ax3.set_xlabel('Item Index')
        ax3.set_ylabel('Value/Weight Ratio')
        ax3.set_title('Value-to-Weight Ratios')
        ax3.grid(True, alpha=0.3)
        
        # 4. Selected items summary
        if len(selected_items) <= 20:  # Only show detailed view for reasonable number of items
            item_names = [f"Item {i}" if 'name' not in item else str(item['name']) 
                         for i, item in enumerate(selected_items)]
            item_values = [item['value'] for item in selected_items]
            
            bars = ax4.barh(item_names, item_values, color='green', alpha=0.6)
            ax4.set_xlabel('Value')
            ax4.set_title(f'Selected Items (Total Value: {total_value:.1f})')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, f'Selected {len(selected_items)} items\nTotal Value: {total_value:.1f}\nTotal Weight: {total_weight:.1f}', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax4.set_title('Solution Summary')
            ax4.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_function_landscape(self, 
                                   func,
                                   bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                                   best_solution: Optional[Tuple[float, float]] = None,
                                   population: Optional[List[Tuple[float, float]]] = None,
                                   title: str = "Function Landscape",
                                   resolution: int = 100,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize 2D function optimization landscape.
        
        Args:
            func: Function to visualize (should accept (x, y) and return fitness)
            bounds: ((x_min, x_max), (y_min, y_max)) bounds for the plot
            best_solution: Best solution point to highlight
            population: Current population points to show
            title: Plot title
            resolution: Grid resolution for landscape
            save_path: Path to save the plot
        """
        (x_min, x_max), (y_min, y_max) = bounds
        
        # Create meshgrid
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function on grid
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                try:
                    Z[i, j] = func((X[i, j], Y[i, j]))
                except:
                    Z[i, j] = np.nan
        
        plt.figure(figsize=self.figsize)
        
        # Plot contour
        contour = plt.contour(X, Y, Z, levels=20, alpha=0.6)
        plt.contourf(X, Y, Z, levels=20, alpha=0.4, cmap='viridis')
        plt.colorbar(label='Fitness')
        
        # Plot population if provided
        if population:
            pop_x = [point[0] for point in population]
            pop_y = [point[1] for point in population]
            plt.scatter(pop_x, pop_y, c='white', s=30, alpha=0.7, 
                       edgecolors='black', linewidth=1, label='Population')
        
        # Plot best solution if provided
        if best_solution:
            plt.scatter(best_solution[0], best_solution[1], c='red', s=100, 
                       marker='*', edgecolors='black', linewidth=2, label='Best Solution')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        if population or best_solution:
            plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def animate_evolution(self, 
                         evolution_data: List[Dict[str, Any]],
                         visualization_func,
                         interval: int = 500,
                         save_path: Optional[str] = None) -> None:
        """
        Create an animation of the evolution process.
        
        Args:
            evolution_data: List of data for each generation
            visualization_func: Function to create visualization for each frame
            interval: Interval between frames in milliseconds
            save_path: Path to save animation (as GIF)
        """
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            print("Animation requires matplotlib.animation. Showing static plots instead.")
            # Show first, middle, and last generations
            indices = [0, len(evolution_data)//2, -1]
            for i, idx in enumerate(indices):
                visualization_func(evolution_data[idx], 
                                 title=f"Generation {idx if idx >= 0 else len(evolution_data)-1}")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def animate(frame):
            ax.clear()
            visualization_func(evolution_data[frame], ax=ax, 
                             title=f"Generation {frame}")
        
        anim = FuncAnimation(fig, animate, frames=len(evolution_data), 
                           interval=interval, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow')
            else:
                anim.save(save_path)
        
        plt.show()