# Examples

This directory contains example scripts demonstrating how to use the various modules in the Tourist Trip Optimizer.

## Available Examples

### `mip_solver_example.py`

Demonstrates how to use the MIP (Mixed Integer Programming) solver module to find optimal tourist routes.

**Usage:**
```bash
cd /home/runner/work/tourist-trip-optimizer/tourist-trip-optimizer
python3 examples/mip_solver_example.py
```

**What it does:**
- Loads tourist attraction data
- Uses a subset of attractions (20) for demonstration
- Shows two ways to use the MIP solver:
  1. Using the `MIPSolver` class directly (for more control)
  2. Using the convenience function `solve_ttdp_mip()` (for quick solutions)
- Prints solution details and verification

**Note:** The MIP solver is computationally intensive. For large datasets (100+ attractions), the solver may take several minutes to find optimal solutions. For quick testing, we recommend using a small subset of attractions.

## Running the Examples

All examples assume you're running them from the repository root directory:

```bash
# Install dependencies first
pip install -r requirements.txt

# Run an example
python3 examples/mip_solver_example.py
```

## Module Usage Patterns

### Pattern 1: Using a Class-based Solver (More Control)

```python
from mip_solver import MIPSolver

solver = MIPSolver(distance_matrix, scores, visit_durations, max_time=24.0)
solver.build_model()
solution = solver.solve(time_limit=300)
solver.print_solution(attractions_df)
```

### Pattern 2: Using Convenience Functions (Quick Solutions)

```python
from mip_solver import solve_ttdp_mip

result = solve_ttdp_mip(distance_matrix, scores, visit_durations, 
                        max_time=24.0, time_limit=300)
print(f"Best tour: {result['tour_sequence']}")
```

## Need Help?

- See the main [README.md](../README.md) for project overview
- See [TECHNICAL.md](../TECHNICAL.md) for algorithm details
- Check the docstrings in each module for detailed API documentation
