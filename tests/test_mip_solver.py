#!/usr/bin/env python3
"""
Integration test for MIP Solver module.

This test verifies that the MIP solver module can be imported and used
successfully with the actual project data.
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
from mip_solver import MIPSolver, solve_ttdp_mip
from data_utils import load_attractions_data

def test_mip_solver_basic():
    """Test basic MIP solver functionality with small dataset."""
    print("Testing MIP Solver Module...")
    print("-" * 60)
    
    # Load test data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    attractions = load_attractions_data(os.path.join(data_dir, 'attractions.csv'))
    distance_matrix = np.load(os.path.join(data_dir, 'distance_matrix.npy'))
    
    # Use small subset for quick test
    n_test = 15
    attractions_test = attractions.head(n_test).copy()
    distance_matrix_test = distance_matrix[:n_test, :n_test]
    
    scores = attractions_test['interest_score'].values
    visit_durations = attractions_test['visit_duration'].values
    
    print(f"✓ Loaded {n_test} attractions for testing")
    
    # Test 1: Import and instantiation
    print("\nTest 1: Create MIPSolver instance")
    solver = MIPSolver(
        distance_matrix=distance_matrix_test,
        scores=scores,
        visit_durations=visit_durations,
        max_time=10.0,
        avg_speed=50.0
    )
    assert solver is not None
    assert solver.n_attractions == n_test
    print("✓ MIPSolver instantiated successfully")
    
    # Test 2: Build model
    print("\nTest 2: Build MIP model")
    solver.build_model()
    assert solver.model is not None
    assert solver.x is not None
    assert solver.y is not None
    assert solver.u is not None
    print("✓ Model built successfully")
    print(f"  Variables: {len(solver.model.variables())}")
    print(f"  Constraints: {len(solver.model.constraints)}")
    
    # Test 3: Solve model
    print("\nTest 3: Solve MIP model")
    solution = solver.solve(time_limit=30, verbose=False)
    assert solution is not None
    assert 'status' in solution
    assert 'objective_value' in solution
    print(f"✓ Model solved: {solution['status']}")
    
    if solution['objective_value'] is not None:
        print(f"  Objective: {solution['objective_value']:.2f}")
        print(f"  Attractions visited: {solution['n_attractions_visited']}")
        
        # Test 4: Verify solution
        print("\nTest 4: Verify solution")
        verification = solver.verify_solution()
        assert 'total_time' in verification
        assert 'total_score' in verification
        assert 'time_constraint_satisfied' in verification
        print(f"✓ Solution verified")
        print(f"  Time used: {verification['total_time']:.2f}/{solver.max_time} hours")
        print(f"  Score: {verification['total_score']:.2f}")
        print(f"  Constraint satisfied: {verification['time_constraint_satisfied']}")
        
        # Test 5: Solution dict method
        print("\nTest 5: Test solution dictionary")
        result = solver.get_solution_dict()
        assert isinstance(result, dict)
        assert 'tour_sequence' in result
        print("✓ Solution dictionary retrieved")
    
    # Test 6: Convenience function
    print("\nTest 6: Test convenience function")
    result2 = solve_ttdp_mip(
        distance_matrix=distance_matrix_test,
        scores=scores,
        visit_durations=visit_durations,
        max_time=10.0,
        time_limit=30,
        verbose=False
    )
    assert result2 is not None
    assert 'status' in result2
    print("✓ Convenience function works")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        test_mip_solver_basic()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
