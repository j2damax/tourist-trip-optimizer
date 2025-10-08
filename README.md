# Tourist Trip Optimizer

A Strategic Framework for Implementing the Tourist Trip Design Problem (TTDP) for Sri Lankan tourist attractions.

## 📊 Comprehensive Optimization Report

**📖 [Complete Optimization Report](OPTIMIZATION_REPORT.md)** - Full academic-style report with literature review, methodology, and analysis

**📈 [Results Summary](RESULTS_SUMMARY.md)** - Quick overview of key findings and recommendations

These reports provide comprehensive analysis of the Tourist Trip Design Problem, including:
- Problem definition with mathematical formulation
- Literature review of 6 related research studies  
- Data exploration and preprocessing (953 Sri Lankan attractions)
- Genetic Algorithm implementation and results (0.24s, score 555.00)
- Mixed Integer Programming implementation (299.89s, score 558.00, optimal)
- Detailed comparison and analysis (GA is 1,249× faster with 0.54% optimality gap)
- Real-world applicability and business impact

## Overview

This project implements optimization algorithms to solve the Tourist Trip Design Problem (TTDP), which aims to maximize tourist satisfaction by selecting and sequencing attractions within time constraints. The implementation includes both a Genetic Algorithm (GA) and a Mixed Integer Programming (MIP) approach for comparison.

### Key Results

- **Genetic Algorithm:** 0.24 seconds, score 555.00 (99.46% of optimal)
- **MIP (Optimal):** 299.89 seconds, score 558.00 (100% optimal, proven)
- **Performance:** GA is 1,249× faster with only 0.54% quality gap
- **Recommendation:** Use GA for real-world applications (scalable, fast, near-optimal)

## Project Structure

```
tourist-trip-optimizer/
├── data/                  # Data directory for datasets and results
│   ├── processed/        # Processed data (attractions, distance matrices)
│   └── results/          # Optimization results (GA, MIP outputs)
├── notebooks/            # Jupyter notebooks for analysis
│   ├── 01_Data_Exploration_and_Preparation.ipynb
│   ├── 02_Genetic_Algorithm_Implementation.ipynb
│   ├── 03_MIP_Model_Benchmark.ipynb
│   └── 04_Results_and_Visualization.ipynb
├── scripts/              # Python modules
│   ├── data_utils.py     # Data loading and processing utilities
│   ├── ga_core.py        # Genetic Algorithm implementation
│   ├── mip_solver.py     # MIP solver implementation
│   └── visualization.py  # Visualization functions
├── examples/            # Usage examples
├── requirements.txt      # Project dependencies
├── OPTIMIZATION_REPORT.md  # 📊 Comprehensive analysis report (full academic report)
├── RESULTS_SUMMARY.md     # 📈 Quick results overview and recommendations
└── README.md            # This file
```

## Features

- **Data Exploration**: Load and analyze Sri Lankan tourist attraction data
- **Genetic Algorithm**: Metaheuristic approach for finding high-quality solutions quickly
- **MIP Model**: Exact optimization using Mixed Integer Programming
- **Visualization**: Interactive maps, charts, and comprehensive reports
- **Comparison**: Side-by-side analysis of different optimization approaches

## Installation

1. Clone the repository:
```bash
git clone https://github.com/j2damax/tourist-trip-optimizer.git
cd tourist-trip-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- pulp: Linear programming and optimization
- haversine: Geographic distance calculations
- matplotlib: Static visualizations
- folium: Interactive map generation

## Usage

### Running the Notebooks

Execute the notebooks in sequence:

1. **01_Data_Exploration_and_Preparation.ipynb**
   - Creates sample Sri Lankan tourist attraction data
   - Explores data statistics and relationships
   - Calculates distance matrices
   - Prepares data for optimization

2. **02_Genetic_Algorithm_Implementation.ipynb**
   - Implements the Genetic Algorithm
   - Runs optimization with configurable parameters
   - Analyzes convergence and solution quality
   - Generates visualizations and reports

3. **03_MIP_Model_Benchmark.ipynb**
   - Formulates the TTDP as a Mixed Integer Program
   - Solves using PuLP's CBC solver
   - Compares results with GA
   - Provides optimality analysis

4. **04_Results_and_Visualization.ipynb**
   - Comprehensive results comparison
   - Interactive maps and visualizations
   - Performance analysis
   - Recommendations and conclusions

### Using the Python Modules

#### Genetic Algorithm

```python
from scripts.data_utils import load_attractions_data, calculate_distance_matrix
from scripts.ga_core import GeneticAlgorithm
from scripts.visualization import plot_route_on_map

# Load data
attractions = load_attractions_data('data/sri_lanka_attractions.csv')
distance_matrix = calculate_distance_matrix(attractions)

# Run Genetic Algorithm
ga = GeneticAlgorithm(
    distance_matrix=distance_matrix,
    scores=attractions['score'].values,
    visit_durations=attractions['visit_duration'].values,
    max_time=24,
    population_size=100,
    generations=500
)

solution, fitness, history = ga.evolve()
tour = ga.get_valid_tour(solution)

# Visualize results
plot_route_on_map(attractions, tour, 'tour_map.html')
```

#### MIP Solver

```python
from scripts.data_utils import load_attractions_data
from scripts.mip_solver import MIPSolver
import numpy as np

# Load data
attractions = load_attractions_data('data/processed/attractions.csv')
distance_matrix = np.load('data/processed/distance_matrix.npy')

# Create and solve MIP model
solver = MIPSolver(
    distance_matrix=distance_matrix,
    scores=attractions['interest_score'].values,
    visit_durations=attractions['visit_duration'].values,
    max_time=24.0,
    avg_speed=50.0
)

solver.build_model()
solution = solver.solve(time_limit=300)  # 5 minute limit
solver.print_solution(attractions)
```

For more examples, see the `examples/` directory.


## Problem Formulation

The Tourist Trip Design Problem is formulated as an Orienteering Problem variant:

**Objective**: Maximize total satisfaction score from visited attractions

**Constraints**:
- Total time (travel + visit) ≤ maximum allowed time
- Each attraction visited at most once
- Valid tour sequence (no subtours)

## Algorithms

### Genetic Algorithm
- Population-based metaheuristic
- Tournament selection
- Order crossover (OX)
- Swap mutation
- Configurable parameters (population size, generations, rates)

### Mixed Integer Programming
- Exact optimization approach
- MTZ subtour elimination constraints
- Solved using CBC solver
- Provides optimal or near-optimal solutions with guarantees

## Results

Both algorithms are benchmarked on solution quality and computation time. The GA typically provides good solutions quickly, while MIP can find optimal solutions but may require more time for larger instances.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Jayampathy Balasuriya

## Acknowledgments

- Tourist attraction data based on popular Sri Lankan destinations
- Algorithms inspired by Operations Research literature on the Orienteering Problem
