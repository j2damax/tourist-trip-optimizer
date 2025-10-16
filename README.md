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
- Requirements: Python, Jupyter Notebook. Install dependencies from `requirements.txt`.

## Structure
```
tourist-trip-optimizer/
├── data/           # Datasets & processed results
├── notebooks/      # Interactive Jupyter Notebooks (analysis, tutorials)
├── scripts/        # Python modules (GA, MIP, utilities, visualization)
├── requirements.txt
```

## Main Results
- GA: ~0.24s, score 555.00 (99.46% optimal)
- MIP: ~299.89s, score 558.00 (100% optimal)
- GA is 1,249× faster, with only 0.54% quality gap

---
**Languages:** Jupyter Notebook (84.6%), Python (15.4%)
**Repo:** [j2damax/tourist-trip-optimizer](https://github.com/j2damax/tourist-trip-optimizer)

> For more details, check out the notebooks and scripts in the repo!