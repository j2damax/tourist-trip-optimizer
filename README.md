# Tourist Trip Optimizer

A Strategic Framework for Implementing the Tourist Trip Design Problem (TTDP) for Sri Lankan tourist attractions.

## What is this project?
This repository provides a practical implementation for solving the Tourist Trip Design Problem (TTDP), which focuses on planning optimal routes for tourists based on preferences and time constraints. It offers two main optimization methods:
- **Genetic Algorithm (GA):** Fast and nearly optimal.
- **Mixed Integer Programming (MIP):** Slow but guarantees optimality.

## Key Features
- Mathematical TTDP formulation and ready-to-use datasets.
- Comprehensive analysis and benchmarking (Sri Lanka dataset: 953 attractions).
- Transparent performance comparison between GA and MIP solvers.
- Jupyter Notebooks for interactive exploration, learning, and visualization.

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/j2damax/tourist-trip-optimizer.git
cd tourist-trip-optimizer

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Try the Demo Notebooks
Start with our easy-to-follow demonstration notebooks:

1. **`genetic-algorithm-demo.ipynb`** - Learn how to use the Genetic Algorithm solver
   - 3 real-world use cases with different trip requirements
   - Cultural Heritage Tour, Nature & Wildlife Adventure, Weekend Getaway
   - Interactive examples you can customize

2. **`mip-demo.ipynb`** - Learn how to use the MIP solver for optimal solutions
   - 3 optimization scenarios (Cultural, Wildlife, Waterfalls)
   - Guaranteed mathematically optimal itineraries
   - Performance comparison with GA

These notebooks demonstrate how to:
- Filter POIs by category, rating, and preferences
- Configure trip parameters (days, hours per day, etc.)
- Run the solvers and analyze results
- Visualize convergence and compare solutions

## Structure
```
tourist-trip-optimizer/
├── data/                          # Datasets & processed results
├── notebooks/                     # Interactive Jupyter Notebooks
│   ├── 01-data-preparation.ipynb
│   ├── 02-genetic-algorithm.ipynb
│   ├── 03-mip-solver.ipynb
│   └── 04-results-analysis.ipynb
├── src/                           # Python modules
│   ├── genetic_algorithm.py       # GA implementation
│   ├── mip_solver.py              # MIP implementation
│   └── utils.py                   # Utility functions
├── genetic-algorithm-demo.ipynb   # 🆕 GA Demo with real use cases
├── mip-demo.ipynb                 # 🆕 MIP Demo with real use cases
├── requirements.txt
└── test_implementation.py         # Test suite
```

## Main Results
- GA: ~0.24s, score 555.00 (99.46% optimal)
- MIP: ~299.89s, score 558.00 (100% optimal)
- GA is 1,249× faster, with only 0.54% quality gap

## Which Solver Should You Use?

### Use Genetic Algorithm (GA) when:
✅ You have many POIs to consider (100+)  
✅ You need results quickly (< 1 second)  
✅ Near-optimal solutions are acceptable (99%+ quality)  
✅ You want to experiment with different parameters  

### Use Mixed-Integer Programming (MIP) when:
✅ You need the absolute best solution  
✅ Working with smaller problem instances (< 50 POIs)  
✅ You can afford longer computation times  
✅ Solution quality must be provable/certified  
✅ Planning high-value or limited-time trips  

---
**Languages:** Jupyter Notebook (84.6%), Python (15.4%)
**Repo:** [j2damax/tourist-trip-optimizer](https://github.com/j2damax/tourist-trip-optimizer)

> For more details, check out the notebooks and scripts in the repo!