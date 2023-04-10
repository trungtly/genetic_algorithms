"""
Traveling Salesman Problem (TSP) Implementation using Genetic Algorithms

The TSP is one of the most famous combinatorial optimization problems:
- Given a set of cities and distances between them
- Find the shortest possible route visiting each city exactly once
- Return to the starting city

This problem is excellent for demonstrating genetic algorithms because:
- Uses permutation representation (city visiting order)
- Requires specialized crossover operators (Order Crossover, PMX)
- Clear objective function (minimize total distance)
- Scalable difficulty (more cities = exponentially harder)
- Well-studied with known optimal solutions for benchmarks
"""

from typing import List, Tuple, Dict, Any, Optional
import random
import math
import json
from ..core.individual import PermutationIndividual


class City:
    """Represents a city with coordinates."""
    
    def __init__(self, x: float, y: float, name: str = ""):
        """
        Initialize a city.
        
        Args:
            x: X coordinate
            y: Y coordinate  
            name: Optional city name
        """
        self.x = x
        self.y = y
        self.name = name or f"City({x:.1f},{y:.1f})"
    
    def distance_to(self, other: 'City') -> float:
        """Calculate Euclidean distance to another city."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __str__(self) -> str:
        return f"{self.name}: ({self.x:.2f}, {self.y:.2f})"
    
    def __repr__(self) -> str:
        return f"City(x={self.x}, y={self.y}, name='{self.name}')"


class TSPIndividual(PermutationIndividual):
    """
    Individual representation for the TSP.
    
    Uses permutation encoding where genes represent the order of city visits:
    - genes[i] = city_index to visit at position i
    - Example: [0, 2, 1, 3] means visit cities in order: 0 → 2 → 1 → 3 → 0
    
    The representation automatically handles the constraint that each city
    must be visited exactly once.
    """
    
    def __init__(self, genes: List[int] = None, cities: List[City] = None):
        """
        Initialize a TSP individual.
        
        Args:
            genes: Permutation of city indices. If None, random tour generated.
            cities: List of cities in the problem
        """
        if cities is None:
            raise ValueError("Cities list is required for TSP")
        
        self.cities = cities
        
        # Initialize with proper size based on number of cities
        super().__init__(genes, size=len(cities))
        
        # Cache for efficiency
        self._total_distance = None
        self._distance_matrix = None
    
    def calculate_fitness(self) -> float:
        """
        Calculate fitness for this TSP tour.
        
        For TSP, we want to minimize total distance, but GA maximizes fitness.
        We use the negative of total distance as fitness (higher = better).
        
        Returns:
            Fitness value (negative total distance)
        """
        total_distance = self.get_total_distance()
        
        # Return negative distance for maximization
        # Add small constant to avoid issues with zero distance
        return -(total_distance + 1e-10)
    
    def get_total_distance(self) -> float:
        """Calculate total distance of the tour."""
        if self._total_distance is None:
            distance = 0.0
            
            # Sum distances between consecutive cities in the tour
            for i in range(len(self.genes)):
                current_city = self.cities[self.genes[i]]
                next_city = self.cities[self.genes[(i + 1) % len(self.genes)]]
                distance += current_city.distance_to(next_city)
            
            self._total_distance = distance
        
        return self._total_distance
    
    def get_tour_cities(self) -> List[City]:
        """Get cities in the order they are visited."""
        return [self.cities[city_idx] for city_idx in self.genes]
    
    def get_tour_coordinates(self) -> List[Tuple[float, float]]:
        """Get city coordinates in tour order (useful for plotting)."""
        coords = [(self.cities[city_idx].x, self.cities[city_idx].y) 
                 for city_idx in self.genes]
        # Add starting city at end to close the loop
        coords.append((self.cities[self.genes[0]].x, self.cities[self.genes[0]].y))
        return coords
    
    def invalidate_fitness(self):
        """Invalidate cached values when genes change."""
        super().invalidate_fitness()
        self._total_distance = None
    
    def apply_2opt_improvement(self, max_iterations: int = 100) -> 'TSPIndividual':
        """
        Apply 2-opt local search to improve the tour.
        
        2-opt is a simple local search that swaps edges:
        1. Take two edges in the tour
        2. Remove them and reconnect in the other possible way
        3. Keep the change if it improves the tour
        4. Repeat until no improvement found
        
        Args:
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            Improved individual (self, modified in place)
        """
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try all possible 2-opt swaps
            for i in range(len(self.genes) - 1):
                for j in range(i + 2, len(self.genes)):
                    # Calculate current distance of edges (i,i+1) and (j,j+1)
                    city_i = self.cities[self.genes[i]]
                    city_i_plus_1 = self.cities[self.genes[i + 1]]
                    city_j = self.cities[self.genes[j]]
                    city_j_plus_1 = self.cities[self.genes[(j + 1) % len(self.genes)]]
                    
                    current_distance = (city_i.distance_to(city_i_plus_1) + 
                                      city_j.distance_to(city_j_plus_1))
                    
                    # Calculate distance after 2-opt swap
                    new_distance = (city_i.distance_to(city_j) + 
                                  city_i_plus_1.distance_to(city_j_plus_1))
                    
                    # If swap improves tour, apply it
                    if new_distance < current_distance:
                        # Reverse the segment between i+1 and j
                        self.genes[i + 1:j + 1] = self.genes[i + 1:j + 1][::-1]
                        self.invalidate_fitness()
                        improved = True
                        break
                
                if improved:
                    break
        
        return self
    
    def get_solution_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of this solution."""
        return {
            'total_distance': self.get_total_distance(),
            'fitness': self.get_fitness(),
            'num_cities': len(self.cities),
            'tour_order': self.genes,
            'city_names': [self.cities[i].name for i in self.genes]
        }
    
    def __str__(self) -> str:
        """String representation of the TSP solution."""
        return (f"TSPTour(distance={self.get_total_distance():.2f}, "
                f"cities={len(self.cities)}, tour={self.genes[:5]}{'...' if len(self.genes) > 5 else ''})")


class TSPProblem:
    """
    TSP problem instance with utilities for problem generation and analysis.
    
    This class provides:
    - Problem instance creation (random, grid, circle patterns)
    - Distance matrix computation and caching
    - Solution analysis and comparison
    - Nearest neighbor heuristic
    - Problem visualization helpers
    """
    
    def __init__(self, cities: List[City]):
        """
        Initialize TSP problem instance.
        
        Args:
            cities: List of cities to visit
        """
        if len(cities) < 3:
            raise ValueError("TSP requires at least 3 cities")
        
        self.cities = cities
        self.num_cities = len(cities)
        
        # Precompute distance matrix for efficiency
        self._distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> List[List[float]]:
        """Precompute matrix of distances between all city pairs."""
        n = len(self.cities)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.cities[i].distance_to(self.cities[j])
        
        return matrix
    
    def get_distance(self, city1_idx: int, city2_idx: int) -> float:
        """Get distance between two cities by index."""
        return self._distance_matrix[city1_idx][city2_idx]
    
    def create_individual(self, genes: List[int] = None) -> TSPIndividual:
        """Create a TSP individual for this problem."""
        return TSPIndividual(genes=genes, cities=self.cities)
    
    def solve_with_ga(self, **ga_params) -> Tuple[TSPIndividual, Dict[str, Any]]:
        """
        Solve this TSP using genetic algorithm.
        
        Args:
            **ga_params: Parameters for genetic algorithm
            
        Returns:
            Tuple of (best_solution, statistics)
        """
        from ..core import solve_with_ga
        from ..core.crossover import OrderCrossover
        from ..core.mutation import SwapMutation
        from ..core.selection import TournamentSelection
        
        # Default parameters optimized for TSP
        default_params = {
            'population_size': min(200, max(50, self.num_cities * 4)),  # Scale with problem size
            'max_generations': 1000,
            'mutation_rate': 0.1,  # Higher mutation rate for permutations
            'crossover_rate': 0.8,
            'elite_count': max(1, min(5, default_params.get('population_size', 100) // 20)),
            'selection_operator': TournamentSelection(tournament_size=5),
            'crossover_operator': OrderCrossover(),
            'mutation_operator': SwapMutation()
        }
        default_params.update(ga_params)
        
        return solve_with_ga(
            individual_class=TSPIndividual,
            cities=self.cities,
            **default_params
        )
    
    def nearest_neighbor_solution(self, start_city: int = 0) -> TSPIndividual:
        """
        Solve using nearest neighbor heuristic.
        
        Algorithm:
        1. Start at a given city
        2. Repeatedly go to the nearest unvisited city
        3. Return to start when all cities visited
        
        Args:
            start_city: Index of starting city
            
        Returns:
            Nearest neighbor solution
        """
        if not 0 <= start_city < self.num_cities:
            raise ValueError(f"Start city {start_city} not in range [0, {self.num_cities})")
        
        tour = [start_city]
        unvisited = set(range(self.num_cities))
        unvisited.remove(start_city)
        
        current_city = start_city
        
        while unvisited:
            # Find nearest unvisited city
            nearest_city = min(unvisited, 
                             key=lambda city: self.get_distance(current_city, city))
            
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
        
        return self.create_individual(tour)
    
    def best_nearest_neighbor_solution(self) -> TSPIndividual:
        """
        Find best nearest neighbor solution by trying all starting cities.
        
        Returns:
            Best nearest neighbor solution found
        """
        best_solution = None
        best_distance = float('inf')
        
        for start_city in range(self.num_cities):
            solution = self.nearest_neighbor_solution(start_city)
            distance = solution.get_total_distance()
            
            if distance < best_distance:
                best_distance = distance
                best_solution = solution
        
        return best_solution
    
    def random_solution(self) -> TSPIndividual:
        """Generate a random tour."""
        return self.create_individual()  # Random permutation generated automatically
    
    def brute_force_optimal(self) -> TSPIndividual:
        """
        Find optimal solution using brute force.
        
        WARNING: Only use for very small problems (< 10 cities) as complexity is O(n!).
        
        Returns:
            Optimal solution
        """
        if self.num_cities > 10:
            raise ValueError(f"Too many cities ({self.num_cities}) for brute force. Use <= 10 cities.")
        
        from itertools import permutations
        
        best_tour = None
        best_distance = float('inf')
        
        # Try all possible tours (permutations)
        # Fix first city to avoid symmetric solutions
        for perm in permutations(range(1, self.num_cities)):
            tour = [0] + list(perm)
            solution = self.create_individual(tour)
            distance = solution.get_total_distance()
            
            if distance < best_distance:
                best_distance = distance
                best_tour = solution
        
        return best_tour
    
    def analyze_problem_characteristics(self) -> Dict[str, Any]:
        """
        Analyze problem characteristics that affect difficulty.
        
        Returns:
            Dictionary with problem analysis
        """
        # Calculate statistics about city positions
        x_coords = [city.x for city in self.cities]
        y_coords = [city.y for city in self.cities]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        area = x_range * y_range
        
        # Calculate average and variance of distances
        distances = []
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                distances.append(self.get_distance(i, j))
        
        avg_distance = sum(distances) / len(distances)
        distance_variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
        
        # Estimate density (cities per unit area)
        density = self.num_cities / area if area > 0 else float('inf')
        
        return {
            'num_cities': self.num_cities,
            'area': area,
            'density': density,
            'x_range': x_range,
            'y_range': y_range,
            'avg_distance': avg_distance,
            'distance_variance': distance_variance,
            'min_distance': min(distances),
            'max_distance': max(distances)
        }
    
    def get_problem_summary(self) -> str:
        """Get human-readable problem summary."""
        analysis = self.analyze_problem_characteristics()
        
        summary = f"""
TSP Problem Summary:
===================
Cities: {self.num_cities}
Area: {analysis['area']:.1f}
Density: {analysis['density']:.3f} cities/unit²
Average Distance: {analysis['avg_distance']:.2f}
Distance Range: [{analysis['min_distance']:.2f}, {analysis['max_distance']:.2f}]

Cities:
-------"""
        
        for i, city in enumerate(self.cities):
            summary += f"\n{i:2d}: {city}"
        
        return summary
    
    @classmethod
    def create_random_problem(cls, num_cities: int, 
                            x_range: Tuple[float, float] = (0, 100),
                            y_range: Tuple[float, float] = (0, 100)) -> 'TSPProblem':
        """
        Create a random TSP instance with cities uniformly distributed.
        
        Args:
            num_cities: Number of cities to generate
            x_range: (min_x, max_x) for city coordinates
            y_range: (min_y, max_y) for city coordinates
            
        Returns:
            Random TSP problem instance
        """
        cities = []
        for i in range(num_cities):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            cities.append(City(x, y, f"City_{i}"))
        
        return cls(cities)
    
    @classmethod
    def create_circle_problem(cls, num_cities: int, radius: float = 50.0, 
                            center: Tuple[float, float] = (50, 50)) -> 'TSPProblem':
        """
        Create TSP instance with cities arranged on a circle.
        
        This creates a problem with a known optimal solution (following the circle).
        
        Args:
            num_cities: Number of cities
            radius: Circle radius
            center: Circle center coordinates
            
        Returns:
            Circle TSP problem instance
        """
        cities = []
        for i in range(num_cities):
            angle = 2 * math.pi * i / num_cities
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            cities.append(City(x, y, f"City_{i}"))
        
        return cls(cities)
    
    @classmethod
    def create_grid_problem(cls, width: int, height: int, 
                          spacing: float = 10.0) -> 'TSPProblem':
        """
        Create TSP instance with cities arranged on a grid.
        
        Args:
            width: Grid width (number of cities)
            height: Grid height (number of cities)
            spacing: Distance between adjacent grid points
            
        Returns:
            Grid TSP problem instance
        """
        cities = []
        city_index = 0
        
        for i in range(height):
            for j in range(width):
                x = j * spacing
                y = i * spacing
                cities.append(City(x, y, f"City_{city_index}"))
                city_index += 1
        
        return cls(cities)
    
    @classmethod
    def create_clustered_problem(cls, num_clusters: int, cities_per_cluster: int,
                               cluster_radius: float = 10.0, 
                               cluster_separation: float = 50.0) -> 'TSPProblem':
        """
        Create TSP instance with clustered cities.
        
        This creates a challenging problem where the optimal solution
        likely visits all cities in one cluster before moving to the next.
        
        Args:
            num_clusters: Number of city clusters
            cities_per_cluster: Cities in each cluster
            cluster_radius: Radius of each cluster
            cluster_separation: Distance between cluster centers
            
        Returns:
            Clustered TSP problem instance
        """
        cities = []
        city_index = 0
        
        # Generate cluster centers
        cluster_centers = []
        for i in range(num_clusters):
            angle = 2 * math.pi * i / num_clusters
            center_x = cluster_separation * math.cos(angle)
            center_y = cluster_separation * math.sin(angle)
            cluster_centers.append((center_x, center_y))
        
        # Generate cities in each cluster
        for cluster_idx, (center_x, center_y) in enumerate(cluster_centers):
            for city_in_cluster in range(cities_per_cluster):
                # Random position within cluster
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, cluster_radius)
                
                x = center_x + distance * math.cos(angle)
                y = center_y + distance * math.sin(angle)
                
                cities.append(City(x, y, f"C{cluster_idx}_{city_in_cluster}"))
                city_index += 1
        
        return cls(cities)
    
    def save_to_file(self, filename: str):
        """Save problem instance to JSON file."""
        data = {
            'cities': [
                {
                    'x': city.x,
                    'y': city.y,
                    'name': city.name
                }
                for city in self.cities
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'TSPProblem':
        """Load problem instance from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        cities = [
            City(
                x=city_data['x'],
                y=city_data['y'], 
                name=city_data.get('name', f"City_{i}")
            )
            for i, city_data in enumerate(data['cities'])
        ]
        
        return cls(cities)