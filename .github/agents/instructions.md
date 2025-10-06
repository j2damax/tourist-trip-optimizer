# GitHub Copilot Agent Instructions - Tourist Trip Optimizer

## Project Context

### Project Name
**Itinerary-Optimizer-SL** - Tourist Trip Design Problem (TTDP) Solution for Sri Lankan Attractions

### Project Mission
Develop and maintain a robust optimization system that generates optimal tourist itineraries for Sri Lankan attractions. The system combines Genetic Algorithms (GA) for scalability with Mixed Integer Programming (MIP) for solution validation, targeting 1000+ points of interest.

### Core Objectives
1. **Scalability**: Handle datasets with 1000+ tourist attractions efficiently
2. **Accuracy**: Provide near-optimal solutions validated against exact methods
3. **Usability**: Deliver intuitive interfaces through Jupyter notebooks and Python APIs
4. **Maintainability**: Write clean, documented, testable code following best practices

## Code Standards and Conventions

### Python Style Guide
- **PEP 8 Compliance**: All Python code must follow PEP 8 guidelines
- **Line Length**: Maximum 100 characters per line (code), 79 for docstrings
- **Indentation**: 4 spaces (no tabs)
- **Naming Conventions**:
  - `snake_case` for functions, variables, and module names
  - `PascalCase` for class names
  - `UPPER_CASE` for constants
  - Descriptive names (avoid abbreviations unless standard: `df` for DataFrame, `GA` for Genetic Algorithm)

### Documentation Requirements

#### Module-level Docstrings
```python
"""
Module Name and Purpose

Brief description of what this module does and its main responsibilities.
Include key classes, functions, and usage patterns.
"""
```

#### Function/Method Docstrings
Use Google-style docstrings:
```python
def function_name(param1, param2):
    """
    Brief description of function purpose.
    
    Longer description if needed, explaining algorithm or approach.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
        
    Returns:
        type: Description of return value
        
    Raises:
        ExceptionType: When this exception occurs
        
    Examples:
        >>> function_name(value1, value2)
        expected_output
    """
```

#### Class Docstrings
```python
class ClassName:
    """
    Brief description of class purpose.
    
    Detailed description of class responsibilities, key attributes,
    and usage patterns.
    
    Attributes:
        attribute1 (type): Description
        attribute2 (type): Description
    """
```

### Code Organization

#### File Structure
```
tourist-trip-optimizer/
├── data/                      # Data files (CSV, NPY)
│   └── .gitignore            # Exclude large data files
├── notebooks/                 # Jupyter notebooks (numbered sequence)
│   ├── 01_Data_Exploration_and_Preparation.ipynb
│   ├── 02_Genetic_Algorithm_Implementation.ipynb
│   ├── 03_MIP_Model_Benchmark.ipynb
│   └── 04_Results_and_Visualization.ipynb
├── scripts/                   # Reusable Python modules
│   ├── __init__.py           # Package initialization
│   ├── data_utils.py         # Data handling utilities
│   ├── ga_core.py            # Genetic Algorithm core
│   ├── mip_solver.py         # MIP formulation and solving
│   └── visualization.py      # Plotting and reporting
├── tests/                     # Unit and integration tests
│   ├── test_data_utils.py
│   ├── test_ga_core.py
│   └── test_visualization.py
├── .github/                   # GitHub configuration
│   └── agents/
│       └── instructions.md   # This file
├── TECHNICAL.md               # Technical documentation
├── TASKS.md                   # Detailed task breakdown
├── README.md                  # User-facing documentation
└── requirements.txt           # Python dependencies
```

#### Module Responsibilities

**scripts/data_utils.py**:
- Load and validate attraction data from CSV
- Calculate distance matrices using haversine formula
- Prepare data structures for optimization
- Data validation and error handling

**scripts/ga_core.py**:
- Implement Genetic Algorithm for TTDP
- Population management
- Genetic operators (selection, crossover, mutation)
- Fitness evaluation
- Tour validation and extraction

**scripts/mip_solver.py** (if created):
- MIP model formulation using PuLP
- Constraint generation
- Solver configuration and execution
- Solution extraction and validation

**scripts/visualization.py**:
- Interactive map generation (Folium)
- Static plots (Matplotlib)
- Fitness evolution charts
- Comparative analysis visualizations
- Summary report generation

### Import Standards

#### Import Order (PEP 8)
1. Standard library imports
2. Related third-party imports
3. Local application imports

```python
# Standard library
import os
import sys
from typing import List, Tuple, Dict

# Third-party
import numpy as np
import pandas as pd
from pulp import LpProblem, LpMaximize

# Local
from scripts.data_utils import load_attractions_data
from scripts.ga_core import GeneticAlgorithm
```

#### Avoid Wildcard Imports
```python
# Bad
from numpy import *

# Good
import numpy as np
```

## Testing Requirements

### Test Structure
- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test module interactions
- **Validation Tests**: Compare GA vs MIP results
- **Performance Tests**: Benchmark algorithm runtime

### Testing Framework
Use `pytest` for all tests:

```python
# tests/test_data_utils.py
import pytest
import numpy as np
import pandas as pd
from scripts.data_utils import calculate_distance_matrix

def test_distance_matrix_symmetry():
    """Test that distance matrix is symmetric."""
    attractions = pd.DataFrame({
        'latitude': [6.9271, 7.2906],
        'longitude': [79.8612, 80.6406],
        'name': ['Colombo', 'Kandy'],
        'score': [8.0, 9.0],
        'visit_duration': [2.0, 3.0]
    })
    
    dist_matrix = calculate_distance_matrix(attractions)
    
    # Check symmetry
    assert np.allclose(dist_matrix, dist_matrix.T)
    
    # Check diagonal is zero
    assert np.allclose(np.diag(dist_matrix), 0)
```

### Test Coverage Goals
- Aim for >80% code coverage
- All public functions must have tests
- Edge cases and error conditions must be tested
- Performance regression tests for key algorithms

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=scripts tests/

# Run specific test file
pytest tests/test_ga_core.py

# Run verbose mode
pytest -v tests/
```

## Common Tasks and Workflows

### Task 1: Adding a New Attraction Data Source

**Steps**:
1. Create data loading function in `data_utils.py`
2. Validate data format (required columns: name, latitude, longitude, score, visit_duration)
3. Add tests for data validation
4. Update documentation

**Example**:
```python
def load_attractions_from_api(api_url, api_key):
    """
    Load attractions from REST API.
    
    Args:
        api_url (str): API endpoint URL
        api_key (str): Authentication key
        
    Returns:
        pd.DataFrame: Validated attraction data
    """
    # Implementation
    data = fetch_api_data(api_url, api_key)
    validated_data = validate_attraction_data(data)
    return validated_data
```

### Task 2: Implementing a New Genetic Operator

**Steps**:
1. Add method to `GeneticAlgorithm` class
2. Ensure operator maintains permutation validity
3. Add parameter to control operator selection
4. Write unit tests
5. Benchmark performance impact
6. Document in TECHNICAL.md

**Example**:
```python
def scramble_mutation(self, individual):
    """
    Scramble mutation: randomly shuffle a substring.
    
    Args:
        individual (list): Tour permutation
        
    Returns:
        list: Mutated tour
    """
    if random.random() < self.mutation_rate:
        i, j = sorted(random.sample(range(len(individual)), 2))
        substring = individual[i:j+1]
        random.shuffle(substring)
        individual[i:j+1] = substring
    return individual
```

### Task 3: Adding a New Visualization

**Steps**:
1. Add function to `visualization.py`
2. Follow consistent plotting style (seaborn-v0_8-darkgrid)
3. Support both interactive (Folium) and static (Matplotlib) outputs
4. Add example usage to docstring
5. Update notebook to include new visualization

**Example**:
```python
def plot_attraction_clusters(attractions_data, cluster_labels, output_file='clusters.html'):
    """
    Visualize attraction clusters on interactive map.
    
    Args:
        attractions_data (pd.DataFrame): Attraction data with coordinates
        cluster_labels (np.ndarray): Cluster assignment for each attraction
        output_file (str): Output HTML file path
    """
    # Create Folium map
    center_lat = attractions_data['latitude'].mean()
    center_lon = attractions_data['longitude'].mean()
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Add markers colored by cluster
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for idx, row in attractions_data.iterrows():
        cluster = cluster_labels[idx]
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['name']} (Cluster {cluster})",
            icon=folium.Icon(color=colors[cluster % len(colors)])
        ).add_to(map_obj)
    
    map_obj.save(output_file)
```

### Task 4: Optimizing Algorithm Performance

**Steps**:
1. Profile code to identify bottlenecks
   ```bash
   python -m cProfile -o profile.stats script.py
   python -m pstats profile.stats
   ```
2. Optimize hot paths (vectorize with NumPy, use caching)
3. Maintain correctness (run tests before/after)
4. Benchmark improvements
5. Document performance gains

**Optimization Patterns**:
```python
# Bad: Python loops
distances = []
for i in range(len(tour)):
    for j in range(len(tour)):
        distances.append(calculate_distance(tour[i], tour[j]))

# Good: Vectorized operations
distances = distance_matrix[np.ix_(tour, tour)]
```

### Task 5: Creating a New Notebook

**Guidelines**:
1. **Naming**: Use numbered prefix (e.g., `05_New_Analysis.ipynb`)
2. **Structure**:
   - Title and description (markdown cell)
   - Import libraries
   - Load data
   - Analysis/implementation
   - Results and visualization
   - Summary and conclusions
3. **Code Quality**: Same standards as scripts
4. **Documentation**: Markdown cells explaining each step
5. **Reproducibility**: Include random seeds, save outputs

**Template**:
```markdown
# Notebook Title

Brief description of notebook purpose and what will be accomplished.

## Import Libraries
[Code cell with imports]

## Load Data
[Code cell loading data]

## Analysis/Implementation
[Multiple sections with code and markdown]

## Results
[Visualizations and metrics]

## Conclusions
[Summary of findings]
```

## Algorithm-Specific Guidelines

### Genetic Algorithm Development

**Key Principles**:
1. **Maintain Permutation Validity**: All operators must preserve valid tours
2. **Configurable Parameters**: Expose all GA parameters (population size, generations, rates)
3. **Fitness Caching**: Consider caching fitness evaluations for efficiency
4. **Diversity Preservation**: Monitor population diversity to avoid premature convergence

**Parameter Tuning**:
```python
# Recommended ranges for parameter tuning
POPULATION_SIZE: [50, 100, 200, 500]
GENERATIONS: [100, 500, 1000]
MUTATION_RATE: [0.01, 0.05, 0.1, 0.2]
CROSSOVER_RATE: [0.7, 0.8, 0.9]
TOURNAMENT_SIZE: [3, 5, 7]
```

### MIP Model Development

**Key Principles**:
1. **Formulation Clarity**: Use descriptive variable names
2. **Constraint Documentation**: Comment each constraint type
3. **Solver Configuration**: Set appropriate time limits and gaps
4. **Solution Validation**: Verify feasibility of extracted solutions

**Best Practices**:
```python
# Create model
model = LpProblem("TTDP", LpMaximize)

# Decision variables with descriptive names
visit = LpVariable.dicts("visit", range(n_attractions), cat='Binary')
travel = LpVariable.dicts("travel", 
                         [(i,j) for i in range(n) for j in range(n) if i!=j],
                         cat='Binary')

# Add objective with comment
# Maximize total satisfaction from visited attractions
model += lpSum([scores[i] * visit[i] for i in range(n_attractions)])

# Add constraints with comments
# Time constraint: total time must not exceed limit
model += lpSum([visit_durations[i] * visit[i] for i in range(n_attractions)]) <= max_time
```

## Data Handling Best Practices

### Data Validation
Always validate input data:
```python
def validate_attraction_data(data):
    """Validate attraction DataFrame has required columns and valid values."""
    required_columns = ['name', 'latitude', 'longitude', 'visit_duration', 'score']
    
    # Check columns exist
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check value ranges
    if not (-90 <= data['latitude']).all() or not (data['latitude'] <= 90).all():
        raise ValueError("Latitude values must be in [-90, 90]")
    
    if not (-180 <= data['longitude']).all() or not (data['longitude'] <= 180).all():
        raise ValueError("Longitude values must be in [-180, 180]")
    
    if not (data['visit_duration'] > 0).all():
        raise ValueError("Visit duration must be positive")
    
    if not (data['score'] >= 0).all():
        raise ValueError("Scores must be non-negative")
    
    return True
```

### Missing Data Handling
```python
# Check for missing values
if data.isnull().any().any():
    # Option 1: Remove rows with missing data
    data_clean = data.dropna()
    
    # Option 2: Impute missing values (if appropriate)
    data_clean = data.fillna({
        'score': data['score'].median(),
        'visit_duration': data['visit_duration'].median()
    })
```

### Large Dataset Handling
For datasets >1000 attractions:
1. Use chunking for distance matrix computation
2. Consider sparse matrix representations
3. Implement progress bars for long operations
4. Save intermediate results

```python
from tqdm import tqdm

def calculate_distance_matrix_chunked(attractions, chunk_size=100):
    """Calculate distance matrix in chunks with progress bar."""
    n = len(attractions)
    distance_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n), desc="Computing distances"):
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

## Error Handling and Logging

### Exception Handling
```python
def load_attractions_data(filepath):
    """Load attraction data with comprehensive error handling."""
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty: {filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")
    
    # Validate data
    if not validate_attraction_data(data):
        raise ValueError("Invalid attraction data format")
    
    return data
```

### Logging (for production code)
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_optimization():
    """Run optimization with logging."""
    logger.info("Starting optimization...")
    
    try:
        # Load data
        logger.info("Loading data...")
        data = load_attractions_data('data/attractions.csv')
        logger.info(f"Loaded {len(data)} attractions")
        
        # Run GA
        logger.info("Initializing Genetic Algorithm...")
        ga = GeneticAlgorithm(...)
        logger.info("Running evolution...")
        solution, fitness, history = ga.evolve()
        logger.info(f"Optimization complete. Best fitness: {fitness}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise
```

## Performance Optimization Guidelines

### Use NumPy for Numerical Operations
```python
# Bad: Python loops
total = 0
for i in range(len(scores)):
    if visit[i]:
        total += scores[i]

# Good: NumPy vectorization
total = np.sum(scores[visit])
```

### Cache Expensive Computations
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def calculate_tour_fitness(tour_tuple):
    """Cache fitness calculations for repeated tours."""
    tour = list(tour_tuple)
    return compute_fitness(tour)
```

### Profile Before Optimizing
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
ga.evolve()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

## Version Control Best Practices

### Commit Messages
Follow conventional commit format:
```
type(scope): brief description

Detailed explanation if needed

- Bullet points for multiple changes
- Reference issue numbers if applicable
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
```
feat(ga): add elitism to genetic algorithm

Implemented elitism by preserving top 10% of population
across generations to prevent loss of best solutions.

- Added elitism_rate parameter to GeneticAlgorithm
- Updated tests to verify elite preservation
- Benchmarked performance improvement

Fixes #42
```

```
fix(data): handle missing coordinates gracefully

Previously crashed on null lat/lon values.
Now logs warning and skips invalid entries.
```

### Branch Naming
- `feature/feature-name`: New features
- `fix/bug-description`: Bug fixes
- `docs/documentation-update`: Documentation only
- `refactor/component-name`: Code refactoring

### Pull Request Guidelines
1. **Description**: Explain what and why
2. **Testing**: Describe tests added/run
3. **Screenshots**: Include for UI/visualization changes
4. **Breaking Changes**: Clearly mark any breaking changes
5. **Checklist**: Code reviewed, tests pass, docs updated

## Debugging Strategies

### Common Issues and Solutions

**Issue**: GA converges prematurely
- Check population diversity
- Increase mutation rate
- Reduce selection pressure
- Implement diversity preservation mechanisms

**Issue**: MIP solver times out
- Reduce problem size (fewer attractions)
- Adjust solver time limit
- Relax optimality gap tolerance
- Consider warm-start with GA solution

**Issue**: Distance matrix seems incorrect
- Verify coordinate order (latitude, longitude)
- Check unit conversions (km vs miles)
- Validate haversine implementation
- Test with known distances

**Issue**: Tour violates time constraints
- Check tour validation logic
- Verify travel time calculations
- Ensure visit durations are correct
- Debug fitness function constraint handling

### Debugging Tools
```python
# Add assertions during development
assert len(tour) == len(set(tour)), "Tour has duplicate attractions"
assert all(0 <= idx < n_attractions for idx in tour), "Invalid indices"

# Use breakpoint for interactive debugging
def fitness(self, individual):
    total_score = 0
    # ...
    breakpoint()  # Drop into debugger here
    # ...

# Verbose logging for debugging
if DEBUG:
    print(f"Generation {gen}: max={max_fit:.2f}, avg={avg_fit:.2f}")
```

## Dependencies and Environment

### Required Dependencies
```txt
# Core scientific computing
numpy>=1.20.0
pandas>=1.3.0

# Optimization
pulp>=2.5.0

# Geographic calculations
haversine>=2.5.0

# Visualization
matplotlib>=3.4.0
folium>=0.12.0

# Notebook support
jupyter>=1.0.0
ipykernel>=6.0.0

# Testing (development)
pytest>=7.0.0
pytest-cov>=3.0.0
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If exists
```

## Code Review Checklist

When reviewing code (or preparing for review):

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have docstrings
- [ ] Type hints are used where appropriate
- [ ] Tests are included and pass
- [ ] No commented-out code (use git history instead)
- [ ] No hardcoded paths or values (use constants/config)
- [ ] Error handling is appropriate
- [ ] Performance is acceptable (no obvious inefficiencies)
- [ ] Documentation is updated (README, TECHNICAL.md)
- [ ] No security issues (no secrets in code)
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Variable names are descriptive
- [ ] Complex logic has explanatory comments

## Getting Help

### Documentation Resources
1. **README.md**: User-facing documentation, installation, basic usage
2. **TECHNICAL.md**: Deep technical details, algorithms, architecture
3. **TASKS.md**: Detailed task breakdown for implementation
4. **Code docstrings**: API documentation within code
5. **Notebooks**: Examples and tutorials

### External Resources
- **NumPy Documentation**: https://numpy.org/doc/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **PuLP Documentation**: https://coin-or.github.io/pulp/
- **Genetic Algorithms**: "Introduction to Genetic Algorithms" by Melanie Mitchell
- **Orienteering Problem**: Vansteenwegen et al. (2011) survey paper

### Asking Questions
When stuck:
1. Check existing documentation first
2. Search codebase for similar patterns
3. Review test files for examples
4. Consult technical literature
5. Ask specific questions with context

## Summary

This project prioritizes:
- **Quality**: Well-tested, documented, maintainable code
- **Performance**: Efficient algorithms, optimized operations
- **Usability**: Clear APIs, helpful error messages, good documentation
- **Reproducibility**: Deterministic results, versioned dependencies

When in doubt, refer to existing code patterns in the repository and maintain consistency with established conventions.

---

**Last Updated**: 2024  
**Maintained By**: Development Team  
**Version**: 1.0
