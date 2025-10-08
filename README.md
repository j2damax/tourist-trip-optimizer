# Tourist Trip Optimizer

A Strategic Framework for Implementing the Tourist Trip Design Problem (TTDP) for Sri Lankan tourist attractions.

> **🎓 New to optimization algorithms?** Start with our [**Comprehensive Learning Guide**](LEARNING_GUIDE.md) – a beginner-friendly deep dive into Genetic Algorithms, MIP, heuristics, and practical optimization.
> **⚡ Just want to run it?** See the [`QUICK_START.md`](QUICK_START.md) to get results in under 15 minutes.

## 📊 Comprehensive Optimization Reports

**📖 Full Report:** [`OPTIMIZATION_REPORT.md`](OPTIMIZATION_REPORT.md) – academic-style methodology, formulation, and analysis.  
**📈 Summary:** [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) – Key metrics & recommendations at a glance.

Highlights:
- Mathematical problem definition & formulation
- Literature review (6 related studies)
- Dataset profiling (953 curated Sri Lankan attractions)
- GA performance: 0.24s, score 555.00 (99.46% optimal)
- MIP performance: 299.89s, score 558.00 (proven optimal)
- GA vs MIP: 1,249× faster with only 0.54% optimality gap
- Business impact & deployment considerations

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
│   ├── 00_Data_Preprocessing_Demo.ipynb
│   ├── 01_Data_Exploration_and_Preparation.ipynb
│   ├── 02_Genetic_Algorithm_Implementation.ipynb
│   ├── 03_MIP_Model_Benchmark.ipynb
│   ├── 04_Results_and_Visualization.ipynb
│   └── 05_Interactive_Learning_Tutorial.ipynb  # 🎓 Interactive learning path
├── scripts/              # Python modules
│   ├── data_utils.py     # Data loading and processing utilities
│   ├── ga_core.py        # Genetic Algorithm implementation
│   ├── mip_solver.py     # MIP solver implementation
│   └── visualization.py  # Visualization functions
├── examples/             # Usage examples (incl. beginner tutorial)
├── LEARNING_GUIDE.md     # 📚 Comprehensive learning guide
├── QUICK_START.md        # 🚀 Minimal run instructions
├── OPTIMIZATION_REPORT.md  # 📊 Full analytical report
├── RESULTS_SUMMARY.md     # 📈 Condensed performance summary
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Features

- **📚 Learning Guide:** 1800+ lines teaching optimization fundamentals
- **🎓 Interactive Tutorial:** Guided notebook for experimentation
- **Data Exploration:** Curated, domain-driven Sri Lankan attractions dataset
- **Genetic Algorithm:** Scalable metaheuristic implementation
- **MIP Model:** Exact benchmark with PuLP (CBC)
- **Visualization:** Fitness curves, geospatial maps, comparative analytics
- **Benchmarking:** Rigorous GA vs MIP comparison workflow

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

### 🎓 For Learners
1. Read sections 1–4 of `LEARNING_GUIDE.md`
2. Run `notebooks/05_Interactive_Learning_Tutorial.ipynb`
3. Explore GA vs MIP in notebooks 02 & 03

### 🚀 Quick Start (GA Only)
See `QUICK_START.md` for a minimal script-based run.

### Notebook Workflow
1. `01_Data_Exploration_and_Preparation.ipynb`
2. `02_Genetic_Algorithm_Implementation.ipynb`
3. `03_MIP_Model_Benchmark.ipynb`
4. `04_Results_and_Visualization.ipynb`
5. (Optional) `05_Interactive_Learning_Tutorial.ipynb`

## Example (Programmatic GA Run)
```python
from scripts.data_utils import load_attractions_data, calculate_distance_matrix
from scripts.ga_core import GeneticAlgorithm

attractions = load_attractions_data('data/processed/attractions.csv')
D = calculate_distance_matrix(attractions)

ga = GeneticAlgorithm(
    distance_matrix=D,
    scores=attractions['interest_score'].values,
    visit_durations=attractions['visit_duration'].values,
    max_time=24,
    population_size=120,
    generations=400
)
solution, fitness, history = ga.evolve()
tour = ga.get_valid_tour(solution)
```

## Example (MIP Benchmark)
```python
import numpy as np
from scripts.mip_solver import solve_ttdp_mip

# Use a subset for tractability if large
scores = attractions['interest_score'].values[:60]
visits = attractions['visit_duration'].values[:60]
D_sub = D[:60, :60]
result = solve_ttdp_mip(D_sub, scores, visits, max_time=24, avg_speed=40, time_limit=300)
print(result['status'], result['objective_value'])
```

## Validation & Comparison
- Use MIP on subsets (≤ 60) to validate GA performance
- Track optimality gap: (MIP - GA) / MIP * 100
- Monitor GA convergence via fitness history

## Roadmap (Selected)
- Elitism + adaptive mutation
- Multi-day itinerary (Team Orienteering extension)
- API + lightweight web interface
- Parameter tuning automation

## Contributing
Pull requests welcome. Open an issue for enhancements or bugs.

## License
MIT License (see `LICENSE`).

## Acknowledgments
Data curation supported by Sri Lankan tourism domain knowledge and field partnerships.

---
For full academic-style narrative see `OPTIMIZATION_REPORT.md`. For learning support see `LEARNING_GUIDE.md`.
