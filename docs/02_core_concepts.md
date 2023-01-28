# Core Concepts of Genetic Algorithms

This document provides a deep dive into the fundamental concepts that make genetic algorithms work. Understanding these concepts is crucial for effectively applying GAs to real-world problems.

## 1. Population and Individuals

### Population Dynamics

The **population** is the collection of all candidate solutions at any given time. Population dynamics significantly affect algorithm performance:

#### Population Size Effects
- **Small populations (10-50)**:
  - Faster per generation
  - Higher risk of premature convergence
  - Limited diversity
  - Good for simple problems

- **Large populations (100-1000+)**:
  - Slower per generation
  - Better exploration
  - Higher diversity
  - Better for complex problems

#### Population Initialization
```python
# Random initialization (most common)
population = [create_random_individual() for _ in range(population_size)]

# Seeded initialization (with prior knowledge)
population = create_diverse_population_with_heuristics()

# Hybrid initialization
half_random = [create_random_individual() for _ in range(pop_size//2)]
half_seeded = [create_good_individual() for _ in range(pop_size//2)]
population = half_random + half_seeded
```

### Individual Representation

The choice of representation (encoding) is critical and problem-dependent:

#### Binary Representation
- **Use case**: Discrete optimization, feature selection
- **Example**: `[1, 0, 1, 1, 0, 0, 1]`
- **Advantages**: Simple operators, well-studied
- **Disadvantages**: May not naturally fit all problems

#### Real-Valued Representation  
- **Use case**: Continuous optimization, parameter tuning
- **Example**: `[1.23, -0.45, 2.67, 0.89]`
- **Advantages**: Natural for numerical problems
- **Disadvantages**: Need specialized operators

#### Permutation Representation
- **Use case**: Ordering problems (TSP, scheduling)
- **Example**: `[3, 1, 4, 2, 5, 0]` (city visiting order)
- **Advantages**: Natural for sequencing problems
- **Disadvantages**: Crossover complexity

#### Tree/Graph Representations
- **Use case**: Genetic programming, neural architecture
- **Advantages**: Can represent complex structures
- **Disadvantages**: Complex operators needed

## 2. Fitness and Selection

### Fitness Function Design

The fitness function is the bridge between your problem and the GA. Good fitness functions are:

#### Characteristics of Good Fitness Functions

1. **Correlated with Objective**: Higher fitness = better solution
2. **Computationally Efficient**: Evaluated many times
3. **Smooth**: Small changes in genes → small changes in fitness
4. **Discriminating**: Can distinguish between solution qualities

#### Common Fitness Function Patterns

```python
# Maximization (standard)
def fitness(individual):
    return objective_value(individual)

# Minimization (convert to maximization)
def fitness(individual):
    cost = cost_function(individual)
    return 1.0 / (1.0 + cost)  # or -cost

# Multi-objective (weighted sum)
def fitness(individual):
    obj1 = objective1(individual)
    obj2 = objective2(individual) 
    return w1 * obj1 + w2 * obj2

# Constrained (penalty method)
def fitness(individual):
    base_fitness = objective(individual)
    penalty = sum(max(0, constraint_violation(individual, c)) 
                  for c in constraints)
    return base_fitness - penalty_weight * penalty
```

### Selection Mechanisms

Selection determines which individuals reproduce. Different methods create different **selection pressure**:

#### Selection Pressure Spectrum
```
Low Pressure ←―――――――――――――――――――――――――――→ High Pressure
Random     Roulette    Rank    Tournament    Truncation
```

#### Tournament Selection (Recommended)
- **Mechanism**: Randomly select k individuals, choose the best
- **Pressure**: Controlled by tournament size k
- **Advantages**: Simple, adjustable pressure, works with negative fitness

```python
def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)
```

#### Roulette Wheel Selection
- **Mechanism**: Probability ∝ fitness
- **Issues**: Sensitive to fitness scaling, doesn't work with negative fitness
- **When to use**: When fitness values are naturally proportional

#### Rank Selection
- **Mechanism**: Probability based on rank, not raw fitness
- **Advantages**: Fitness-scale independent
- **Disadvantages**: Loses fitness magnitude information

## 3. Crossover (Recombination)

Crossover combines genetic material from parents to create offspring. It's the primary **exploitation** operator.

### Crossover Design Principles

1. **Heritability**: Offspring should inherit traits from parents
2. **Validity**: Offspring should be valid solutions
3. **Diversity**: Should create new combinations
4. **Locality**: Small changes in parents → small changes in offspring

### Binary Crossover Operators

#### Single-Point Crossover
```
Parent1: [1,1,0|0,1,1,0]    Point at position 3
Parent2: [0,0,1|1,0,0,1]
         ―――――|―――――――
Child1:  [1,1,0|1,0,0,1]
Child2:  [0,0,1|0,1,1,0]
```

#### Two-Point Crossover
```
Parent1: [1,1|0,0,1|1,0]    Points at 2 and 5
Parent2: [0,0|1,1,0|0,1]
         ――|―――――|――
Child1:  [1,1|1,1,0|1,0]
Child2:  [0,0|0,0,1|0,1]
```

#### Uniform Crossover
```
Parent1: [1,1,0,0,1,1,0]
Parent2: [0,0,1,1,0,0,1]
Mask:    [1,0,1,0,0,1,1]    1=take from parent1, 0=from parent2
Child1:  [1,0,0,1,0,1,0]
```

### Real-Valued Crossover

#### Arithmetic Crossover
```python
# Weighted average
child1 = α * parent1 + (1-α) * parent2
child2 = (1-α) * parent1 + α * parent2
```

#### Blend Crossover (BLX-α)
```python
# Extend range by α, then sample uniformly
for each gene:
    min_val = min(parent1[i], parent2[i])
    max_val = max(parent1[i], parent2[i])
    range_size = max_val - min_val
    
    extended_min = min_val - α * range_size
    extended_max = max_val + α * range_size
    
    child1[i] = random.uniform(extended_min, extended_max)
```

### Permutation Crossover

#### Order Crossover (OX)
```
Parent1: [1,2,3|4,5,6|7,8]    Segment: 4,5,6
Parent2: [2,4,6,8,7,5,3,1]
         ―――――|―――――|―――――

Step 1: Copy segment to child
Child1:  [_,_,_,4,5,6,_,_]

Step 2: Fill remaining positions with Parent2 order
Parent2 order: 2,4,6,8,7,5,3,1  (skip 4,5,6 already used)
Remaining:     2,8,7,3,1
Child1:  [2,8,7,4,5,6,3,1]
```

## 4. Mutation

Mutation introduces random variation and is primarily an **exploration** operator.

### Mutation Principles

1. **Diversity Maintenance**: Prevents population convergence
2. **Local Search**: Small changes for fine-tuning
3. **Innovation**: Introduces completely new genetic material
4. **Balance**: Too little → convergence, too much → random search

### Mutation Rates

The mutation rate is typically:
- **Per gene**: 0.001 to 0.1 (1 in 1000 to 1 in 10 genes mutated)
- **Per individual**: Often set so ~1 gene per individual mutates on average

```python
# Rule of thumb for per-gene mutation rate
mutation_rate = 1.0 / chromosome_length
```

### Binary Mutation
```python
def bit_flip_mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip bit
```

### Real-Valued Mutation

#### Gaussian Mutation
```python
def gaussian_mutation(individual, mutation_rate, sigma=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += random.gauss(0, sigma)
```

#### Polynomial Mutation (Self-Adaptive)
- Uses polynomial probability distribution
- Automatically respects bounds
- Higher probability of small changes

### Permutation Mutation

#### Swap Mutation
```python
def swap_mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
```

#### Insertion Mutation
```python
def insertion_mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        i = random.randint(0, len(individual)-1)
        j = random.randint(0, len(individual)-1)
        element = individual.pop(i)
        individual.insert(j, element)
```

## 5. Replacement Strategies

How to form the next generation from parents and offspring:

### Generational Replacement
- Replace entire population with offspring
- Simple but can lose good solutions
- Often combined with elitism

### Steady-State Replacement
- Replace only a few individuals per generation
- Better preservation of good solutions
- More computationally efficient

### Elitism
- Always preserve the best N individuals
- Prevents loss of best solutions
- Typically N = 1-5% of population size

```python
def elitist_replacement(parents, offspring, elite_count):
    # Sort parents by fitness
    parents.sort(key=lambda x: x.fitness, reverse=True)
    elite = parents[:elite_count]
    
    # Combine elite with best offspring
    offspring.sort(key=lambda x: x.fitness, reverse=True)
    next_generation = elite + offspring[:population_size - elite_count]
    
    return next_generation
```

## 6. Algorithm Parameters

### Critical Parameters

1. **Population Size**
   - Rule of thumb: 50-200 for most problems
   - Larger for more complex problems
   - Consider computational budget

2. **Crossover Rate**
   - Typical range: 0.6-0.9
   - Higher values → more exploitation
   - Problem-dependent optimum

3. **Mutation Rate**
   - Per-gene: ~1/chromosome_length
   - Per-individual: ~1.0
   - Adaptive approaches often work well

4. **Selection Pressure**
   - Tournament size: 2-7
   - Higher pressure → faster convergence
   - Risk of premature convergence

### Parameter Interaction Effects

Parameters don't work in isolation:

- **High crossover + Low mutation**: Good for exploitation
- **Low crossover + High mutation**: More exploration
- **High selection pressure + High mutation**: Balanced
- **Low selection pressure + Low mutation**: May converge slowly

## 7. Convergence and Termination

### Convergence Indicators

1. **Fitness Stagnation**: Best fitness hasn't improved for many generations
2. **Population Diversity Loss**: All individuals become very similar
3. **Target Fitness Reached**: Problem-specific goal achieved

### Termination Criteria

```python
def should_terminate(ga):
    # Maximum generations
    if ga.generation >= max_generations:
        return True
    
    # Target fitness reached
    if ga.best_fitness >= target_fitness:
        return True
    
    # Stagnation detection
    if len(ga.fitness_history) > 50:
        recent_improvement = (ga.fitness_history[-1] - 
                            ga.fitness_history[-50])
        if recent_improvement < 1e-6:
            return True
    
    # Diversity collapse
    if ga.population_diversity < 1e-6:
        return True
    
    return False
```

## 8. Common Pitfalls and Solutions

### Premature Convergence
**Problem**: Population converges to suboptimal solution too quickly
**Solutions**:
- Increase population size
- Reduce selection pressure
- Increase mutation rate
- Use diversity preservation techniques

### Slow Convergence
**Problem**: Algorithm takes too long to find good solutions
**Solutions**:
- Increase selection pressure
- Use better initialization
- Tune crossover/mutation balance
- Consider hybrid approaches

### Loss of Diversity
**Problem**: All individuals become too similar
**Solutions**:
- Maintain minimum mutation rate
- Use fitness sharing or crowding
- Implement diversity metrics
- Restart with new random individuals

### Fitness Scaling Issues
**Problem**: Fitness differences too small/large for effective selection
**Solutions**:
- Use rank-based selection
- Apply fitness scaling (linear, power, etc.)
- Design better fitness function

## 9. Performance Metrics

### Convergence Metrics
- **Best fitness over time**: Track improvement
- **Average fitness**: Population quality
- **Fitness variance**: Population diversity
- **Success rate**: Percentage of runs finding good solutions

### Efficiency Metrics
- **Evaluations to solution**: How many fitness evaluations needed
- **Time to convergence**: Wall-clock time
- **Scalability**: Performance vs. problem size

### Diversity Metrics
```python
def population_diversity(population):
    # Genotypic diversity (Hamming distance for binary)
    total_distance = 0
    count = 0
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            distance = hamming_distance(population[i], population[j])
            total_distance += distance
            count += 1
    return total_distance / count if count > 0 else 0

def fitness_diversity(population):
    # Phenotypic diversity (fitness standard deviation)
    fitness_values = [ind.fitness for ind in population]
    return statistics.stdev(fitness_values)
```

## Next Steps

Now that you understand the core concepts, you're ready to:

1. **Implement your own GA**: See [Implementation Guide](03_implementation.md)
2. **Study specific problems**: Explore [Example Problems](04_examples.md)  
3. **Learn advanced techniques**: Read [Advanced Topics](05_advanced.md)

The key to success with genetic algorithms is understanding how these concepts interact and affect your specific problem. Experimentation and careful analysis of results are essential for mastering GA design and application.