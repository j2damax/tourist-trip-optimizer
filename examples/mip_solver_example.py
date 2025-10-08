"""
Example: Using the MIP Solver Module

This script demonstrates how to use the standalone MIP solver module
to solve the Tourist Trip Design Problem (TTDP).
"""

import sys
sys.path.append('scripts')

import numpy as np
from data_utils import load_attractions_data
from mip_solver import MIPSolver, solve_ttdp_mip

# Load attraction data
print("Loading data...")
attractions = load_attractions_data('data/processed/attractions.csv')
distance_matrix = np.load('data/processed/distance_matrix.npy')

# For demonstration, use a subset of attractions (MIP is computationally intensive)
n_subset = 20  # Use first 20 attractions for faster solving
attractions_subset = attractions.head(n_subset).copy()
distance_matrix_subset = distance_matrix[:n_subset, :n_subset]

scores = attractions_subset['interest_score'].values
visit_durations = attractions_subset['visit_duration'].values

print(f"Using {n_subset} attractions for demonstration")

# Example 1: Using the MIPSolver class directly
print("\n" + "="*60)
print("Example 1: Using MIPSolver class")
print("="*60)

solver = MIPSolver(
    distance_matrix=distance_matrix_subset,
    scores=scores,
    visit_durations=visit_durations,
    max_time=12.0,  # 12 hours available
    avg_speed=50.0  # 50 km/h average speed
)

# Build the optimization model
solver.build_model()

# Solve the model
solution = solver.solve(time_limit=60, verbose=False)  # 1 minute time limit

# Print results
solver.print_solution(attractions_subset)

# Verify the solution
verification = solver.verify_solution()
print("\nVerification:")
print(f"  Total time used: {verification['total_time']:.2f} hours")
print(f"  Time constraint satisfied: {verification['time_constraint_satisfied']}")

# Example 2: Using the convenience function
print("\n" + "="*60)
print("Example 2: Using convenience function solve_ttdp_mip()")
print("="*60)

result = solve_ttdp_mip(
    distance_matrix=distance_matrix_subset,
    scores=scores,
    visit_durations=visit_durations,
    max_time=12.0,
    avg_speed=50.0,
    time_limit=60,
    verbose=False  # Suppress solver output
)

print(f"\nResult status: {result['status']}")
print(f"Best score: {result['objective_value']}")
print(f"Tour length: {result['n_attractions_visited']} attractions")
print(f"Tour sequence: {result['tour_sequence']}")

print("\n" + "="*60)
print("Examples completed!")
print("="*60)
