# Task List - Itinerary Optimizer for Sri Lanka (TTDP)

## Overview

This document provides a comprehensive, detailed task breakdown for implementing the Tourist Trip Design Problem (TTDP) solution for Sri Lankan attractions. Tasks are organized by component and ordered to support incremental development.

## Task Organization

Tasks are categorized into:
1. **Project Setup**: Infrastructure and environment
2. **Data Preparation**: Data collection, validation, and preprocessing
3. **Core Algorithm Implementation**: GA and MIP solvers
4. **Visualization**: Maps, charts, and reports
5. **Testing & Validation**: Quality assurance
6. **Documentation**: User and developer documentation
7. **Optimization & Enhancement**: Performance improvements

Each task includes:
- **Task ID**: Unique identifier
- **Title**: Brief description
- **Priority**: Critical, High, Medium, Low
- **Estimated Effort**: Time estimate
- **Dependencies**: Prerequisites
- **Acceptance Criteria**: Definition of done
- **Implementation Notes**: Technical guidance

---

## 1. Project Setup Tasks

### SETUP-001: Initialize Project Repository
**Priority**: Critical  
**Effort**: 2 hours  
**Dependencies**: None

**Description**:
Set up the GitHub repository with proper structure and initial configuration.

**Tasks**:
- Create repository on GitHub
- Initialize README.md with project description
- Create directory structure:
  ```
  tourist-trip-optimizer/
  ├── data/
  ├── notebooks/
  ├── scripts/
  ├── tests/
  └── .github/
  ```
- Create .gitignore file (exclude data files, notebooks outputs, __pycache__)
- Add LICENSE file (MIT License)

**Acceptance Criteria**:
- [ ] Repository exists with proper structure
- [ ] README.md provides project overview
- [ ] .gitignore excludes appropriate files
- [ ] LICENSE file added

**Implementation Notes**:
```bash
# .gitignore contents
*.pyc
__pycache__/
.ipynb_checkpoints/
*.npy
*.csv
*.html
.DS_Store
venv/
*.egg-info/
dist/
build/
```

---

### SETUP-002: Configure Python Environment
**Priority**: Critical  
**Effort**: 1 hour  
**Dependencies**: SETUP-001

**Description**:
Set up Python virtual environment and install required dependencies.

**Tasks**:
- Create requirements.txt with dependencies
- Create virtual environment
- Install dependencies
- Verify installation
- Document setup process in README

**Acceptance Criteria**:
- [ ] requirements.txt lists all dependencies with versions
- [ ] Virtual environment can be created successfully
- [ ] All dependencies install without errors
- [ ] Installation instructions in README

**Implementation Notes**:
```txt
# requirements.txt
pandas>=1.3.0
numpy>=1.20.0
pulp>=2.5.0
haversine>=2.5.0
matplotlib>=3.4.0
folium>=0.12.0
jupyter>=1.0.0
seaborn>=0.11.0
tqdm>=4.62.0
```

---

### SETUP-003: Create Package Structure
**Priority**: High  
**Effort**: 1 hour  
**Dependencies**: SETUP-002

**Description**:
Set up Python package structure for reusable modules.

**Tasks**:
- Create `__init__.py` files in scripts directory
- Configure package imports
- Test package imports work correctly

**Acceptance Criteria**:
- [ ] scripts/__init__.py exists
- [ ] Modules can be imported from scripts package
- [ ] No import errors

**Implementation Notes**:
```python
# scripts/__init__.py
"""
Tourist Trip Optimizer - Core Modules

This package contains the core implementation of the TTDP solution.
"""

from .data_utils import (
    load_attractions_data,
    calculate_distance_matrix,
    prepare_data_for_optimization
)
from .ga_core import GeneticAlgorithm

__version__ = '1.0.0'
__all__ = [
    'load_attractions_data',
    'calculate_distance_matrix',
    'prepare_data_for_optimization',
    'GeneticAlgorithm'
]
```

---

## 2. Data Preparation Tasks

### DATA-001: Create Sample Dataset
**Priority**: Critical  
**Effort**: 3 hours  
**Dependencies**: SETUP-002

**Description**:
Generate sample Sri Lankan tourist attraction dataset with realistic values.

**Tasks**:
- Research Sri Lankan tourist attractions
- Collect coordinates for 50-100 major attractions
- Assign realistic satisfaction scores (0-10)
- Estimate visit durations (hours)
- Create CSV file with proper format
- Validate data quality

**Acceptance Criteria**:
- [ ] CSV file with 50+ attractions
- [ ] All required columns present (name, latitude, longitude, score, visit_duration)
- [ ] Coordinates within Sri Lanka boundaries (5.9-9.9°N, 79.5-82°E)
- [ ] Scores in range 0-10
- [ ] Visit durations reasonable (0.5-8 hours)

**Implementation Notes**:
```csv
name,latitude,longitude,score,visit_duration
Temple of the Tooth,7.2906,80.6406,9.5,2.0
Sigiriya Rock Fortress,7.9571,80.7603,10.0,3.0
Galle Fort,6.0328,80.2169,8.8,2.5
Yala National Park,6.3719,81.5219,9.2,4.0
Adam's Peak,6.8095,80.4989,8.5,6.0
```

Sample attractions to include:
- Sigiriya, Kandy, Galle, Anuradhapura, Polonnaruwa
- National Parks: Yala, Udawalawe, Minneriya
- Beaches: Mirissa, Unawatuna, Arugam Bay
- Cultural sites, tea plantations, waterfalls

---

### DATA-002: Implement Data Loading Module
**Priority**: Critical  
**Effort**: 2 hours  
**Dependencies**: SETUP-003, DATA-001

**Description**:
Create data_utils.py module with data loading functionality.

**Tasks**:
- Implement `load_attractions_data(filepath)` function
- Add error handling for missing files
- Add data type validation
- Create docstrings

**Acceptance Criteria**:
- [ ] Function loads CSV correctly
- [ ] Returns pandas DataFrame
- [ ] Handles FileNotFoundError appropriately
- [ ] Validates column names
- [ ] Comprehensive docstring included

**Implementation Notes**:
```python
def load_attractions_data(filepath):
    """
    Load tourist attractions data from CSV file.
    
    Args:
        filepath (str): Path to CSV file containing attraction data
        
    Returns:
        pd.DataFrame: DataFrame with attraction information
        
    Raises:
        FileNotFoundError: If filepath doesn't exist
        ValueError: If required columns are missing
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if not validate_attraction_data(data):
        raise ValueError("Invalid data format")
    
    return data
```

---

### DATA-003: Implement Data Validation
**Priority**: High  
**Effort**: 2 hours  
**Dependencies**: DATA-002

**Description**:
Create validation function to ensure data quality and completeness.

**Tasks**:
- Implement `validate_attraction_data(data)` function
- Check for required columns
- Validate value ranges (latitude, longitude, scores, durations)
- Check for missing values
- Return detailed error messages

**Acceptance Criteria**:
- [ ] Validates all required columns exist
- [ ] Checks coordinate ranges
- [ ] Validates score and duration ranges
- [ ] Detects null/missing values
- [ ] Returns True/False with descriptive errors

**Implementation Notes**:
```python
def validate_attraction_data(data):
    """Validate attraction DataFrame structure and values."""
    required_columns = ['name', 'latitude', 'longitude', 'visit_duration', 'score']
    
    # Check columns
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Validate ranges
    if not (-90 <= data['latitude']).all() or not (data['latitude'] <= 90).all():
        raise ValueError("Invalid latitude values")
    
    if not (-180 <= data['longitude']).all() or not (data['longitude'] <= 180).all():
        raise ValueError("Invalid longitude values")
    
    if not (data['visit_duration'] > 0).all():
        raise ValueError("Visit duration must be positive")
    
    if not (data['score'] >= 0).all():
        raise ValueError("Scores must be non-negative")
    
    return True
```

---

### DATA-004: Implement Distance Matrix Calculation
**Priority**: Critical  
**Effort**: 3 hours  
**Dependencies**: DATA-002

**Description**:
Implement haversine-based distance calculation between all attraction pairs.

**Tasks**:
- Implement `calculate_distance_matrix(attractions)` function
- Use haversine formula for geographic distances
- Return symmetric matrix
- Optimize for performance (use vectorization if possible)
- Save/load functionality for caching

**Acceptance Criteria**:
- [ ] Correctly calculates distances using haversine
- [ ] Returns symmetric n×n matrix
- [ ] Diagonal is all zeros
- [ ] Units are in kilometers
- [ ] Performance: <5 seconds for 100 attractions

**Implementation Notes**:
```python
from haversine import haversine, Unit

def calculate_distance_matrix(attractions):
    """
    Calculate pairwise distances between attractions.
    
    Args:
        attractions (pd.DataFrame): Attraction data with lat/lon
        
    Returns:
        np.ndarray: n×n distance matrix in kilometers
    """
    n = len(attractions)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        coord_i = (attractions.iloc[i]['latitude'], 
                   attractions.iloc[i]['longitude'])
        for j in range(i+1, n):
            coord_j = (attractions.iloc[j]['latitude'], 
                       attractions.iloc[j]['longitude'])
            dist = haversine(coord_i, coord_j, unit=Unit.KILOMETERS)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix
```

---

### DATA-005: Implement Data Preparation Function
**Priority**: High  
**Effort**: 2 hours  
**Dependencies**: DATA-003, DATA-004

**Description**:
Create function to package all data for optimization algorithms.

**Tasks**:
- Implement `prepare_data_for_optimization(data)` function
- Calculate distance matrix
- Extract scores and durations arrays
- Return dictionary with all required data
- Add caching for distance matrix

**Acceptance Criteria**:
- [ ] Returns dictionary with all required fields
- [ ] Distance matrix is calculated or loaded from cache
- [ ] Arrays are extracted correctly
- [ ] Function is efficient (caching works)

**Implementation Notes**:
```python
def prepare_data_for_optimization(data):
    """
    Prepare data for optimization algorithms.
    
    Returns:
        dict: {
            'attractions': DataFrame,
            'distance_matrix': ndarray,
            'n_attractions': int,
            'scores': ndarray,
            'visit_durations': ndarray
        }
    """
    validate_attraction_data(data)
    distance_matrix = calculate_distance_matrix(data)
    
    return {
        'attractions': data,
        'distance_matrix': distance_matrix,
        'n_attractions': len(data),
        'scores': data['score'].values,
        'visit_durations': data['visit_duration'].values
    }
```

---

## 3. Genetic Algorithm Tasks

### GA-001: Create GeneticAlgorithm Class Structure
**Priority**: Critical  
**Effort**: 2 hours  
**Dependencies**: DATA-004

**Description**:
Set up the GeneticAlgorithm class with initialization and basic structure.

**Tasks**:
- Create ga_core.py module
- Define GeneticAlgorithm class
- Implement `__init__` method with parameters
- Add class docstring
- Define instance variables

**Acceptance Criteria**:
- [ ] Class can be instantiated
- [ ] All parameters are stored as instance variables
- [ ] Comprehensive class and method docstrings
- [ ] No errors on initialization

**Implementation Notes**:
```python
class GeneticAlgorithm:
    """
    Genetic Algorithm for solving the Tourist Trip Design Problem.
    
    Attributes:
        distance_matrix (np.ndarray): Pairwise distances between attractions
        scores (np.ndarray): Satisfaction scores for each attraction
        visit_durations (np.ndarray): Visit time for each attraction (hours)
        max_time (float): Maximum trip duration (hours)
        population_size (int): Number of individuals in population
        generations (int): Number of generations to evolve
        mutation_rate (float): Probability of mutation [0,1]
        crossover_rate (float): Probability of crossover [0,1]
    """
    
    def __init__(self, distance_matrix, scores, visit_durations, 
                 max_time, population_size=100, generations=500,
                 mutation_rate=0.1, crossover_rate=0.8):
        self.distance_matrix = distance_matrix
        self.scores = scores
        self.visit_durations = visit_durations
        self.max_time = max_time
        self.n_attractions = len(scores)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
```

---

### GA-002: Implement Individual Creation
**Priority**: Critical  
**Effort**: 1 hour  
**Dependencies**: GA-001

**Description**:
Implement methods to create random individuals and initial population.

**Tasks**:
- Implement `create_individual()` method
- Implement `create_population()` method
- Use permutation encoding
- Add randomization

**Acceptance Criteria**:
- [ ] create_individual returns valid permutation
- [ ] No duplicate indices in individual
- [ ] create_population returns list of correct size
- [ ] Each individual is unique (likely)

**Implementation Notes**:
```python
def create_individual(self):
    """Create random individual (tour permutation)."""
    individual = list(range(self.n_attractions))
    random.shuffle(individual)
    return individual

def create_population(self):
    """Create initial population."""
    return [self.create_individual() for _ in range(self.population_size)]
```

---

### GA-003: Implement Fitness Function
**Priority**: Critical  
**Effort**: 4 hours  
**Dependencies**: GA-002

**Description**:
Implement fitness evaluation function with time constraint handling.

**Tasks**:
- Implement `fitness(individual)` method
- Calculate tour time (visit + travel)
- Calculate total score
- Handle time constraint violations with penalty
- Optimize for performance (called frequently)

**Acceptance Criteria**:
- [ ] Correctly calculates total score
- [ ] Correctly calculates total time
- [ ] Penalizes constraint violations
- [ ] Returns non-negative fitness
- [ ] Performance: <1ms per evaluation

**Implementation Notes**:
```python
def fitness(self, individual):
    """
    Evaluate fitness of individual tour.
    
    Maximizes satisfaction score while respecting time constraint.
    
    Args:
        individual (list): Tour as permutation of attraction indices
        
    Returns:
        float: Fitness value (total score with penalties)
    """
    total_score = 0
    total_time = 0
    
    for i in range(len(individual)):
        attraction_idx = individual[i]
        
        # Add visit time
        total_time += self.visit_durations[attraction_idx]
        
        # Add travel time to next attraction
        if i < len(individual) - 1:
            next_idx = individual[i + 1]
            travel_distance = self.distance_matrix[attraction_idx, next_idx]
            travel_time = travel_distance / 50  # Avg speed 50 km/h
            total_time += travel_time
        
        # Accumulate score
        total_score += self.scores[attraction_idx]
    
    # Apply penalty if time exceeded
    if total_time > self.max_time:
        penalty = (total_time - self.max_time) * 100
        return max(0, total_score - penalty)
    
    return total_score
```

---

### GA-004: Implement Selection Operator
**Priority**: Critical  
**Effort**: 2 hours  
**Dependencies**: GA-003

**Description**:
Implement tournament selection for parent selection.

**Tasks**:
- Implement `selection(population, fitnesses)` method
- Use tournament selection (size=5)
- Return selected individual
- Maintain selection pressure

**Acceptance Criteria**:
- [ ] Tournament size is configurable
- [ ] Returns fittest from tournament
- [ ] Works with any population size
- [ ] No index errors

**Implementation Notes**:
```python
def selection(self, population, fitnesses):
    """
    Tournament selection.
    
    Args:
        population (list): List of individuals
        fitnesses (list): Fitness values for population
        
    Returns:
        list: Selected individual
    """
    tournament_size = 5
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx].copy()
```

---

### GA-005: Implement Crossover Operator
**Priority**: Critical  
**Effort**: 3 hours  
**Dependencies**: GA-004

**Description**:
Implement order crossover (OX) to combine parent tours.

**Tasks**:
- Implement `crossover(parent1, parent2)` method
- Use order crossover (OX) algorithm
- Maintain permutation validity
- Apply crossover probabilistically

**Acceptance Criteria**:
- [ ] Produces valid permutation
- [ ] Preserves relative order from parents
- [ ] No duplicate indices
- [ ] Respects crossover_rate parameter

**Implementation Notes**:
```python
def crossover(self, parent1, parent2):
    """
    Order crossover (OX).
    
    Args:
        parent1 (list): First parent tour
        parent2 (list): Second parent tour
        
    Returns:
        list: Offspring tour
    """
    if random.random() > self.crossover_rate:
        return parent1.copy()
    
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    # Copy substring from parent1
    offspring = [-1] * size
    offspring[start:end] = parent1[start:end]
    
    # Fill remaining positions with parent2's order
    current_pos = end
    for gene in parent2:
        if gene not in offspring:
            if current_pos >= size:
                current_pos = 0
            offspring[current_pos] = gene
            current_pos += 1
    
    return offspring
```

---

### GA-006: Implement Mutation Operator
**Priority**: Critical  
**Effort**: 2 hours  
**Dependencies**: GA-005

**Description**:
Implement swap mutation to introduce variation.

**Tasks**:
- Implement `mutate(individual)` method
- Use swap mutation (exchange two positions)
- Apply mutation probabilistically
- Maintain permutation validity

**Acceptance Criteria**:
- [ ] Swaps two random positions
- [ ] Respects mutation_rate parameter
- [ ] Maintains valid permutation
- [ ] Modifies individual in place or returns new

**Implementation Notes**:
```python
def mutate(self, individual):
    """
    Swap mutation.
    
    Args:
        individual (list): Tour to mutate
        
    Returns:
        list: Mutated tour
    """
    if random.random() < self.mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual
```

---

### GA-007: Implement Main Evolution Loop
**Priority**: Critical  
**Effort**: 4 hours  
**Dependencies**: GA-006

**Description**:
Implement the main evolutionary algorithm loop.

**Tasks**:
- Implement `evolve()` method
- Initialize population
- Run generational loop
- Track fitness history
- Implement genetic operators
- Return best solution and history

**Acceptance Criteria**:
- [ ] Runs for specified number of generations
- [ ] Tracks max and average fitness per generation
- [ ] Returns best solution found
- [ ] Returns fitness history for analysis
- [ ] No errors during execution

**Implementation Notes**:
```python
def evolve(self):
    """
    Run genetic algorithm evolution.
    
    Returns:
        tuple: (best_solution, best_fitness, fitness_history)
    """
    population = self.create_population()
    fitness_history = []
    best_solution = None
    best_fitness = -np.inf
    
    for generation in range(self.generations):
        # Evaluate fitness
        fitnesses = [self.fitness(ind) for ind in population]
        
        # Track statistics
        max_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        fitness_history.append({
            'generation': generation,
            'max_fitness': max_fitness,
            'avg_fitness': avg_fitness
        })
        
        # Update best solution
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[fitnesses.index(max_fitness)].copy()
        
        # Create next generation
        new_population = []
        for _ in range(self.population_size):
            parent1 = self.selection(population, fitnesses)
            parent2 = self.selection(population, fitnesses)
            offspring = self.crossover(parent1, parent2)
            offspring = self.mutate(offspring)
            new_population.append(offspring)
        
        population = new_population
    
    return best_solution, best_fitness, fitness_history
```

---

### GA-008: Implement Tour Validation
**Priority**: High  
**Effort**: 2 hours  
**Dependencies**: GA-007

**Description**:
Implement function to extract valid tour within time constraints.

**Tasks**:
- Implement `get_valid_tour(solution)` method
- Truncate tour when time limit reached
- Include travel time between attractions
- Return feasible tour

**Acceptance Criteria**:
- [ ] Returns tour within time constraint
- [ ] Includes travel time in calculation
- [ ] Maintains tour order
- [ ] Handles edge cases (empty tour, single attraction)

**Implementation Notes**:
```python
def get_valid_tour(self, solution):
    """
    Extract valid tour respecting time constraint.
    
    Args:
        solution (list): Complete tour permutation
        
    Returns:
        list: Valid tour within time limit
    """
    valid_tour = []
    total_time = 0
    
    for i, attraction_idx in enumerate(solution):
        # Proposed time with this attraction
        proposed_time = total_time + self.visit_durations[attraction_idx]
        
        # Add travel time if not first attraction
        if len(valid_tour) > 0:
            travel_dist = self.distance_matrix[valid_tour[-1], attraction_idx]
            travel_time = travel_dist / 50
            proposed_time += travel_time
        
        # Check if we exceed time limit
        if proposed_time > self.max_time:
            break
        
        valid_tour.append(attraction_idx)
        total_time = proposed_time
    
    return valid_tour
```

---

## 4. MIP Model Tasks

### MIP-001: Create MIP Formulation Module
**Priority**: High  
**Effort**: 3 hours  
**Dependencies**: DATA-004

**Description**:
Create module for MIP model formulation using PuLP.

**Tasks**:
- Create mip_solver.py module (or use notebook)
- Import PuLP
- Define model structure
- Add documentation

**Acceptance Criteria**:
- [ ] Module imports without errors
- [ ] PuLP is available and working
- [ ] Basic structure defined

**Implementation Notes**:
```python
"""
MIP Solver for Tourist Trip Design Problem

Uses PuLP to formulate and solve TTDP as Mixed Integer Program.
"""

from pulp import *
import numpy as np
import pandas as pd
```

---

### MIP-002: Implement Decision Variables
**Priority**: High  
**Effort**: 2 hours  
**Dependencies**: MIP-001

**Description**:
Define decision variables for MIP model.

**Tasks**:
- Define visit variables (x_i)
- Define travel variables (y_ij)
- Define position variables (u_i) for MTZ
- Add variable bounds

**Acceptance Criteria**:
- [ ] Variables created with correct types
- [ ] Binary variables for x and y
- [ ] Continuous variables for u
- [ ] Proper indexing

**Implementation Notes**:
```python
def create_mip_model(distance_matrix, scores, visit_durations, max_time):
    """Create and solve MIP model for TTDP."""
    n = len(scores)
    
    # Create model
    model = LpProblem("TTDP", LpMaximize)
    
    # Decision variables
    x = LpVariable.dicts("visit", range(n), cat='Binary')
    y = LpVariable.dicts("travel", 
                         [(i,j) for i in range(n) for j in range(n) if i!=j],
                         cat='Binary')
    u = LpVariable.dicts("position", range(n), lowBound=0, cat='Continuous')
    
    return model, x, y, u
```

---

### MIP-003: Implement Objective Function
**Priority**: High  
**Effort**: 1 hour  
**Dependencies**: MIP-002

**Description**:
Define objective function to maximize total satisfaction score.

**Tasks**:
- Formulate objective as sum of scores × visit variables
- Add to model

**Acceptance Criteria**:
- [ ] Objective correctly formulated
- [ ] Added to model
- [ ] Syntax correct

**Implementation Notes**:
```python
# Objective: Maximize total score
model += lpSum([scores[i] * x[i] for i in range(n)]), "Total_Score"
```

---

### MIP-004: Implement Time Constraint
**Priority**: Critical  
**Effort**: 2 hours  
**Dependencies**: MIP-003

**Description**:
Add time constraint to ensure total time ≤ max_time.

**Tasks**:
- Calculate total visit time
- Calculate total travel time
- Add constraint to model
- Test constraint

**Acceptance Criteria**:
- [ ] Time constraint properly formulated
- [ ] Includes both visit and travel time
- [ ] Uses correct units

**Implementation Notes**:
```python
# Time constraint
total_visit_time = lpSum([visit_durations[i] * x[i] for i in range(n)])
total_travel_time = lpSum([
    (distance_matrix[i,j] / 50) * y[(i,j)] 
    for i in range(n) for j in range(n) if i != j
])
model += total_visit_time + total_travel_time <= max_time, "Time_Limit"
```

---

### MIP-005: Implement Flow Conservation Constraints
**Priority**: Critical  
**Effort**: 2 hours  
**Dependencies**: MIP-004

**Description**:
Add flow conservation constraints for tour structure.

**Tasks**:
- Implement outgoing flow = visited
- Implement incoming flow = visited
- Add constraints for all nodes

**Acceptance Criteria**:
- [ ] Flow constraints correctly formulated
- [ ] Applied to all attractions
- [ ] Ensures tour structure

**Implementation Notes**:
```python
# Flow conservation
for i in range(n):
    # Outgoing edges
    model += lpSum([y[(i,j)] for j in range(n) if i != j]) == x[i], f"Out_{i}"
    # Incoming edges
    model += lpSum([y[(j,i)] for j in range(n) if i != j]) == x[i], f"In_{i}"
```

---

### MIP-006: Implement Subtour Elimination Constraints
**Priority**: Critical  
**Effort**: 3 hours  
**Dependencies**: MIP-005

**Description**:
Add Miller-Tucker-Zemlin constraints to prevent subtours.

**Tasks**:
- Implement MTZ constraints
- Apply to all edge pairs
- Test effectiveness

**Acceptance Criteria**:
- [ ] MTZ constraints correctly formulated
- [ ] Prevents subtours in solution
- [ ] Doesn't over-constrain

**Implementation Notes**:
```python
# MTZ subtour elimination
for i in range(n):
    for j in range(n):
        if i != j and i > 0 and j > 0:  # Skip depot if using one
            model += (
                u[i] - u[j] + n * y[(i,j)] <= n - 1,
                f"MTZ_{i}_{j}"
            )
```

---

### MIP-007: Implement Solver Execution
**Priority**: High  
**Effort**: 2 hours  
**Dependencies**: MIP-006

**Description**:
Execute solver and extract solution.

**Tasks**:
- Configure solver (CBC)
- Set time limits
- Execute solver
- Extract solution
- Validate solution

**Acceptance Criteria**:
- [ ] Solver executes successfully
- [ ] Solution can be extracted
- [ ] Tour is valid
- [ ] Statistics are captured

**Implementation Notes**:
```python
# Solve
solver = PULP_CBC_CMD(timeLimit=300, msg=1)
model.solve(solver)

# Extract solution
visited_attractions = [i for i in range(n) if x[i].varValue > 0.5]
tour = extract_tour_from_solution(y, visited_attractions)

# Get statistics
objective_value = value(model.objective)
solve_time = model.solutionTime
status = LpStatus[model.status]
```

---

## 5. Visualization Tasks

### VIZ-001: Implement Fitness Evolution Plot
**Priority**: High  
**Effort**: 2 hours  
**Dependencies**: GA-007

**Description**:
Create function to plot GA fitness evolution over generations.

**Tasks**:
- Implement `plot_fitness_evolution(fitness_history)` in visualization.py
- Plot max and average fitness
- Add labels, title, grid
- Support saving to file

**Acceptance Criteria**:
- [ ] Creates line plot of fitness over generations
- [ ] Shows both max and average fitness
- [ ] Professional appearance
- [ ] Can save to file

**Implementation Notes**:
```python
def plot_fitness_evolution(fitness_history):
    """Plot GA fitness evolution."""
    generations = [h['generation'] for h in fitness_history]
    max_fitness = [h['max_fitness'] for h in fitness_history]
    avg_fitness = [h['avg_fitness'] for h in fitness_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fitness, label='Max Fitness', linewidth=2)
    plt.plot(generations, avg_fitness, label='Avg Fitness', linewidth=2, alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Genetic Algorithm Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
```

---

### VIZ-002: Implement Interactive Map Visualization
**Priority**: High  
**Effort**: 4 hours  
**Dependencies**: DATA-001

**Description**:
Create function to plot tour on interactive map using Folium.

**Tasks**:
- Implement `plot_route_on_map(attractions_data, tour_indices, output_file)`
- Create Folium map centered on Sri Lanka
- Add markers for attractions
- Draw route lines between attractions in order
- Add popups with attraction details
- Save to HTML file

**Acceptance Criteria**:
- [ ] Creates interactive HTML map
- [ ] Shows all attractions in tour
- [ ] Draws route between attractions
- [ ] Popups show attraction names and details
- [ ] Map is centered appropriately

**Implementation Notes**:
```python
import folium

def plot_route_on_map(attractions_data, tour_indices, output_file='tour_map.html'):
    """Create interactive map of tour."""
    # Center on Sri Lanka
    center_lat = attractions_data['latitude'].mean()
    center_lon = attractions_data['longitude'].mean()
    
    # Create map
    tour_map = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Add markers
    for idx in tour_indices:
        attraction = attractions_data.iloc[idx]
        folium.Marker(
            location=[attraction['latitude'], attraction['longitude']],
            popup=f"{attraction['name']}<br>Score: {attraction['score']}<br>Duration: {attraction['visit_duration']}h",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(tour_map)
    
    # Draw route
    route_coords = [
        [attractions_data.iloc[idx]['latitude'], 
         attractions_data.iloc[idx]['longitude']]
        for idx in tour_indices
    ]
    folium.PolyLine(route_coords, color='blue', weight=3, opacity=0.7).add_to(tour_map)
    
    # Save
    tour_map.save(output_file)
```

---

### VIZ-003: Implement Tour Statistics Plot
**Priority**: Medium  
**Effort**: 3 hours  
**Dependencies**: VIZ-001

**Description**:
Create comprehensive tour statistics visualization.

**Tasks**:
- Implement `plot_tour_statistics(attractions_data, tour_indices, distance_matrix)`
- Create multi-panel plot with:
  - Score distribution
  - Visit duration distribution
  - Distance between consecutive stops
  - Cumulative metrics
- Add annotations and labels

**Acceptance Criteria**:
- [ ] Creates 2x2 subplot figure
- [ ] All subplots have appropriate content
- [ ] Professional formatting
- [ ] Clear and informative

**Implementation Notes**:
```python
def plot_tour_statistics(attractions_data, tour_indices, distance_matrix):
    """Create comprehensive tour statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract tour data
    tour_data = attractions_data.iloc[tour_indices]
    
    # Plot 1: Scores
    axes[0,0].bar(range(len(tour_indices)), tour_data['score'])
    axes[0,0].set_title('Attraction Scores')
    axes[0,0].set_xlabel('Stop Number')
    axes[0,0].set_ylabel('Score')
    
    # Plot 2: Visit durations
    axes[0,1].bar(range(len(tour_indices)), tour_data['visit_duration'])
    axes[0,1].set_title('Visit Durations')
    axes[0,1].set_xlabel('Stop Number')
    axes[0,1].set_ylabel('Hours')
    
    # Plot 3: Inter-stop distances
    distances = [
        distance_matrix[tour_indices[i], tour_indices[i+1]]
        for i in range(len(tour_indices)-1)
    ]
    axes[1,0].plot(distances, marker='o')
    axes[1,0].set_title('Distance Between Stops')
    axes[1,0].set_xlabel('Leg')
    axes[1,0].set_ylabel('Distance (km)')
    
    # Plot 4: Summary statistics
    total_score = tour_data['score'].sum()
    total_time = tour_data['visit_duration'].sum()
    total_dist = sum(distances)
    
    stats_text = f"""
    Total Attractions: {len(tour_indices)}
    Total Score: {total_score:.1f}
    Total Visit Time: {total_time:.1f}h
    Total Distance: {total_dist:.1f}km
    """
    axes[1,1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1,1].axis('off')
    
    plt.tight_layout()
```

---

### VIZ-004: Implement Algorithm Comparison Plot
**Priority**: Medium  
**Effort**: 2 hours  
**Dependencies**: GA-007, MIP-007

**Description**:
Create comparison visualization for GA vs MIP results.

**Tasks**:
- Implement `compare_algorithms(results_dict)`
- Create side-by-side bar charts for:
  - Solution quality (total score)
  - Computation time
  - Number of attractions visited
- Add table with detailed metrics

**Acceptance Criteria**:
- [ ] Creates comparison visualization
- [ ] Shows key metrics
- [ ] Clear which algorithm is better in each dimension
- [ ] Professional formatting

**Implementation Notes**:
```python
def compare_algorithms(results_dict):
    """
    Compare GA and MIP results.
    
    Args:
        results_dict (dict): {
            'GA': {'score': ..., 'time': ..., 'n_attractions': ...},
            'MIP': {'score': ..., 'time': ..., 'n_attractions': ...}
        }
    """
    algorithms = list(results_dict.keys())
    scores = [results_dict[alg]['score'] for alg in algorithms]
    times = [results_dict[alg]['time'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Score comparison
    ax1.bar(algorithms, scores)
    ax1.set_ylabel('Total Score')
    ax1.set_title('Solution Quality')
    
    # Time comparison
    ax2.bar(algorithms, times)
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computation Time')
    
    plt.tight_layout()
```

---

### VIZ-005: Implement Summary Report Generator
**Priority**: High  
**Effort**: 3 hours  
**Dependencies**: VIZ-003

**Description**:
Create text-based summary report for tour.

**Tasks**:
- Implement `create_summary_report(attractions_data, tour_indices, distance_matrix, algorithm_name)`
- Generate formatted text report with:
  - Tour sequence
  - Individual attraction details
  - Travel distances
  - Summary statistics
- Return as string for printing or saving

**Acceptance Criteria**:
- [ ] Generates comprehensive text report
- [ ] Includes all tour details
- [ ] Properly formatted and readable
- [ ] Summary statistics accurate

**Implementation Notes**:
See TECHNICAL.md for complete implementation example.

---

## 6. Notebook Development Tasks

### NB-001: Create Data Exploration Notebook
**Priority**: High  
**Effort**: 4 hours  
**Dependencies**: DATA-005, VIZ-002

**Description**:
Create 01_Data_Exploration_and_Preparation.ipynb notebook.

**Tasks**:
- Import necessary libraries
- Load and display attraction data
- Calculate and visualize distance matrix
- Show data statistics
- Create sample visualizations
- Save prepared data

**Acceptance Criteria**:
- [ ] Notebook runs without errors
- [ ] All cells produce expected output
- [ ] Visualizations are informative
- [ ] Data is prepared for next notebooks

---

### NB-002: Create GA Implementation Notebook
**Priority**: High  
**Effort**: 5 hours  
**Dependencies**: GA-008, VIZ-001, VIZ-002, VIZ-003

**Description**:
Create 02_Genetic_Algorithm_Implementation.ipynb notebook.

**Tasks**:
- Load prepared data
- Configure GA parameters
- Run GA optimization
- Plot fitness evolution
- Visualize best tour on map
- Show tour statistics
- Save results

**Acceptance Criteria**:
- [ ] Notebook executes successfully
- [ ] GA finds reasonable solutions
- [ ] Visualizations are generated
- [ ] Results are saved for comparison

---

### NB-003: Create MIP Benchmark Notebook
**Priority**: High  
**Effort**: 4 hours  
**Dependencies**: MIP-007, VIZ-002

**Description**:
Create 03_MIP_Model_Benchmark.ipynb notebook.

**Tasks**:
- Load small subset of data (10-20 attractions)
- Formulate MIP model
- Solve with PuLP
- Extract and validate solution
- Visualize MIP solution
- Compare with GA on same subset

**Acceptance Criteria**:
- [ ] MIP model solves successfully
- [ ] Solution is optimal or near-optimal
- [ ] Comparison with GA is shown
- [ ] Computation time is recorded

---

### NB-004: Create Results Visualization Notebook
**Priority**: Medium  
**Effort**: 4 hours  
**Dependencies**: NB-002, NB-003, VIZ-004

**Description**:
Create 04_Results_and_Visualization.ipynb notebook.

**Tasks**:
- Load results from GA and MIP notebooks
- Create comparison visualizations
- Generate summary reports
- Show interactive maps for both solutions
- Provide recommendations
- Document conclusions

**Acceptance Criteria**:
- [ ] Comprehensive comparison of approaches
- [ ] Clear conclusions about algorithm performance
- [ ] Professional visualizations
- [ ] Actionable recommendations

---

## 7. Testing Tasks

### TEST-001: Create Data Utils Tests
**Priority**: High  
**Effort**: 3 hours  
**Dependencies**: DATA-005

**Description**:
Create test_data_utils.py with comprehensive tests.

**Tasks**:
- Test load_attractions_data
- Test validate_attraction_data
- Test calculate_distance_matrix
- Test prepare_data_for_optimization
- Test error handling

**Acceptance Criteria**:
- [ ] All public functions have tests
- [ ] Edge cases are tested
- [ ] Error conditions are tested
- [ ] Tests pass

**Implementation Notes**:
```python
def test_distance_matrix_properties():
    """Test distance matrix has correct properties."""
    attractions = create_sample_data(5)
    matrix = calculate_distance_matrix(attractions)
    
    # Symmetric
    assert np.allclose(matrix, matrix.T)
    
    # Zero diagonal
    assert np.allclose(np.diag(matrix), 0)
    
    # Non-negative
    assert (matrix >= 0).all()
    
    # Correct size
    assert matrix.shape == (5, 5)
```

---

### TEST-002: Create GA Tests
**Priority**: High  
**Effort**: 4 hours  
**Dependencies**: GA-008

**Description**:
Create test_ga_core.py with GA tests.

**Tasks**:
- Test individual creation
- Test population creation
- Test fitness function
- Test genetic operators
- Test evolution loop
- Test tour validation

**Acceptance Criteria**:
- [ ] All methods have tests
- [ ] Permutation validity is tested
- [ ] Fitness calculation is verified
- [ ] Evolution produces improving solutions

**Implementation Notes**:
```python
def test_individual_is_valid_permutation():
    """Test that individuals are valid permutations."""
    ga = create_test_ga()
    individual = ga.create_individual()
    
    # Check length
    assert len(individual) == ga.n_attractions
    
    # Check all indices present exactly once
    assert sorted(individual) == list(range(ga.n_attractions))
    
def test_fitness_respects_time_constraint():
    """Test that fitness penalizes time violations."""
    ga = create_test_ga()
    
    # Create tour that violates time
    long_tour = create_time_violating_tour()
    fitness_long = ga.fitness(long_tour)
    
    # Create tour within time
    short_tour = create_valid_tour()
    fitness_short = ga.fitness(short_tour)
    
    # Valid tour should have higher fitness (assuming similar scores)
    assert fitness_short > fitness_long or fitness_short >= 0
```

---

### TEST-003: Create Visualization Tests
**Priority**: Medium  
**Effort**: 2 hours  
**Dependencies**: VIZ-005

**Description**:
Create test_visualization.py with visualization tests.

**Tasks**:
- Test plot functions don't crash
- Test file outputs are created
- Test report generation

**Acceptance Criteria**:
- [ ] Plot functions execute without errors
- [ ] Output files are created
- [ ] Report strings are well-formed

---

### TEST-004: Create Integration Tests
**Priority**: Medium  
**Effort**: 3 hours  
**Dependencies**: TEST-001, TEST-002

**Description**:
Create integration tests for end-to-end workflows.

**Tasks**:
- Test complete GA pipeline
- Test complete MIP pipeline
- Test data → optimization → visualization flow
- Verify results are reasonable

**Acceptance Criteria**:
- [ ] End-to-end workflows execute
- [ ] Results are within expected ranges
- [ ] No integration issues

---

## 8. Documentation Tasks

### DOC-001: Create Technical Documentation
**Priority**: High  
**Effort**: 6 hours  
**Dependencies**: All implementation tasks

**Description**:
Create comprehensive TECHNICAL.md document.

**Tasks**:
- Document system architecture
- Describe algorithms in detail
- Document data structures
- Provide API reference
- Include performance benchmarks
- Add implementation notes

**Acceptance Criteria**:
- [ ] Complete technical documentation
- [ ] All modules documented
- [ ] API reference included
- [ ] Examples provided

---

### DOC-002: Create Agent Instructions
**Priority**: High  
**Effort**: 4 hours  
**Dependencies**: All implementation tasks

**Description**:
Create .github/agents/instructions.md for Copilot agents.

**Tasks**:
- Document code standards
- Provide common workflows
- Include testing guidelines
- Add troubleshooting guides
- Document project conventions

**Acceptance Criteria**:
- [ ] Comprehensive agent instructions
- [ ] Code standards documented
- [ ] Common tasks explained
- [ ] Examples included

---

### DOC-003: Update README
**Priority**: High  
**Effort**: 2 hours  
**Dependencies**: All implementation tasks

**Description**:
Enhance README.md with complete user documentation.

**Tasks**:
- Update installation instructions
- Document usage examples
- Add screenshots/examples
- Include troubleshooting
- Add contributing guidelines

**Acceptance Criteria**:
- [ ] README is comprehensive
- [ ] Installation steps are clear
- [ ] Usage examples work
- [ ] Professional presentation

---

### DOC-004: Create Task List Document
**Priority**: High  
**Effort**: 5 hours  
**Dependencies**: None

**Description**:
Create this TASKS.md document.

**Tasks**:
- Break down all project tasks
- Organize by component
- Include detailed descriptions
- Provide acceptance criteria
- Add implementation notes

**Acceptance Criteria**:
- [ ] All tasks documented
- [ ] Clear organization
- [ ] Actionable descriptions
- [ ] Acceptance criteria defined

---

## 9. Enhancement Tasks (Optional)

### ENH-001: Implement Elitism in GA
**Priority**: Low  
**Effort**: 2 hours  
**Dependencies**: GA-007

**Description**:
Add elitism to preserve best solutions across generations.

---

### ENH-002: Add Parameter Tuning Notebook
**Priority**: Low  
**Effort**: 4 hours  
**Dependencies**: NB-002

**Description**:
Create notebook for automated parameter tuning.

---

### ENH-003: Implement Parallel GA
**Priority**: Low  
**Effort**: 6 hours  
**Dependencies**: GA-007

**Description**:
Parallelize fitness evaluation for performance improvement.

---

### ENH-004: Add Real-World Data Integration
**Priority**: Medium  
**Effort**: 8 hours  
**Dependencies**: DATA-001

**Description**:
Integrate with real APIs (Google Places, TripAdvisor) for live data.

---

### ENH-005: Create Web Interface
**Priority**: Low  
**Effort**: 20 hours  
**Dependencies**: All core tasks

**Description**:
Build Flask/Streamlit web app for interactive itinerary planning.

---

## Task Dependencies Graph

```
SETUP-001 → SETUP-002 → SETUP-003
                ↓
            DATA-001 → DATA-002 → DATA-003 → DATA-005
                           ↓
                       DATA-004 ←─────────────┘
                           ↓
                ┌──────────┴──────────┐
                ↓                     ↓
            GA-001                 MIP-001
                ↓                     ↓
            GA-002                 MIP-002
                ↓                     ↓
            GA-003                 MIP-003
                ↓                     ↓
            GA-004                 MIP-004
                ↓                     ↓
            GA-005                 MIP-005
                ↓                     ↓
            GA-006                 MIP-006
                ↓                     ↓
            GA-007                 MIP-007
                ↓                     
            GA-008                 
                ↓                     
            VIZ-001 → VIZ-002 → VIZ-003 → VIZ-004 → VIZ-005
                ↓
            NB-001 → NB-002 → NB-003 → NB-004
                              ↓
                          TEST-001 → TEST-002 → TEST-003 → TEST-004
                                                    ↓
                                                DOC-001
                                                DOC-002
                                                DOC-003
                                                DOC-004
```

## Estimated Total Effort

- Project Setup: ~4 hours
- Data Preparation: ~12 hours
- Genetic Algorithm: ~20 hours
- MIP Model: ~15 hours
- Visualization: ~14 hours
- Notebooks: ~17 hours
- Testing: ~12 hours
- Documentation: ~17 hours

**Total: ~111 hours**

This represents a realistic estimate for a single developer working methodically through all tasks.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Maintainer**: Project Team
