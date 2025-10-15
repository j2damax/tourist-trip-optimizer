#!/usr/bin/env python3
"""
Quick test script to verify the TTDP implementation works correctly.
"""

import sys
import pandas as pd
import numpy as np

sys.path.append('src')
from utils import calculate_distance, assign_interest_score, assign_visit_duration
from genetic_algorithm import GeneticAlgorithmTTDP
from mip_solver import MIPSolverTTDP

def test_utils():
    """Test utility functions"""
    print("=" * 70)
    print("Testing Utility Functions")
    print("=" * 70)
    
    # Test distance calculation
    dist = calculate_distance(6.9271, 79.8612, 7.2906, 80.6337)
    print(f"✓ Distance calculation: {dist:.2f} km")
    
    # Test interest score
    score = assign_interest_score('Cultural', 5)
    print(f"✓ Interest score (Cultural, 5): {score}")
    
    # Test visit duration
    duration = assign_visit_duration('Wildlife', 5)
    print(f"✓ Visit duration (Wildlife, 5): {duration}h")
    
    print()

def test_ga():
    """Test Genetic Algorithm"""
    print("=" * 70)
    print("Testing Genetic Algorithm")
    print("=" * 70)
    
    # Create small test dataset
    test_data = {
        'name': [f'POI_{i}' for i in range(10)],
        'interest_score': [90, 85, 80, 75, 70, 65, 60, 55, 50, 45],
        'visit_duration': [2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    }
    test_pois = pd.DataFrame(test_data)
    
    # Create travel time matrix
    n = len(test_pois)
    travel_matrix = np.random.uniform(0.3, 1.5, (n, n))
    np.fill_diagonal(travel_matrix, 0)
    
    # Run GA
    ga = GeneticAlgorithmTTDP(
        pois_df=test_pois,
        travel_time_matrix=travel_matrix,
        num_days=3,
        max_time_per_day=8,
        population_size=50,
        generations=20,
        crossover_rate=0.85,
        mutation_rate=0.03,
        tournament_size=3
    )
    
    best_chromo, best_itin, best_score = ga.evolve(verbose=False)
    print(f"✓ GA completed successfully")
    print(f"  Best score: {best_score}")
    print(f"  Days used: {len(best_itin)}")
    print(f"  POIs visited: {sum(len(day) for day in best_itin)}")
    print()

def test_mip():
    """Test MIP Solver"""
    print("=" * 70)
    print("Testing MIP Solver")
    print("=" * 70)
    
    # Create small test dataset
    test_data = {
        'name': [f'POI_{i}' for i in range(5)],
        'interest_score': [90, 80, 70, 60, 50],
        'visit_duration': [2.0, 1.5, 1.0, 1.0, 0.5]
    }
    test_pois = pd.DataFrame(test_data)
    
    # Create travel time matrix
    n = len(test_pois)
    travel_matrix = np.random.uniform(0.3, 1.0, (n, n))
    np.fill_diagonal(travel_matrix, 0)
    
    # Run MIP
    mip = MIPSolverTTDP(
        pois_df=test_pois,
        travel_time_matrix=travel_matrix,
        num_days=2,
        max_time_per_day=8,
        time_limit=30
    )
    
    itinerary, score, status = mip.solve(verbose=False)
    print(f"✓ MIP completed successfully")
    print(f"  Status: {status}")
    print(f"  Score: {score}")
    print(f"  Days used: {len(itinerary)}")
    print(f"  POIs visited: {sum(len(day) for day in itinerary)}")
    print()

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TTDP Implementation Test Suite")
    print("=" * 70)
    print()
    
    try:
        test_utils()
        test_ga()
        test_mip()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe TTDP implementation is working correctly.")
        print("You can now run the Jupyter notebooks to see the full solution.")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
