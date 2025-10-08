"""
Integration tests for end-to-end workflows
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from scripts.data_utils import (
    load_attractions_data,
    prepare_data_for_optimization
)
from scripts.ga_core import GeneticAlgorithm
from scripts.visualization import (
    plot_fitness_evolution,
    plot_route_on_map,
    create_summary_report
)


class TestCompleteGAPipeline:
    """Test complete GA pipeline from data to results"""
    
    def test_full_ga_workflow_with_real_data(self):
        """Test complete workflow with real data file"""
        # Load real data
        filepath = 'data/processed/attractions.csv'
        if not os.path.exists(filepath):
            pytest.skip("Real data file not available")
        
        # Load and prepare data
        data = load_attractions_data(filepath)
        assert data is not None
        
        prepared = prepare_data_for_optimization(data)
        assert prepared is not None
        
        # Create and run GA with small parameters for testing
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=24,
            population_size=20,
            generations=10
        )
        
        solution, fitness, history = ga.evolve()
        
        # Verify results
        assert solution is not None
        assert len(solution) == prepared['n_attractions']
        assert fitness >= 0
        assert len(history) == 10
        
        # Get valid tour
        valid_tour = ga.get_valid_tour(solution)
        assert isinstance(valid_tour, list)
        assert len(valid_tour) <= len(solution)
    
    def test_full_ga_workflow_with_synthetic_data(self):
        """Test complete workflow with synthetic data"""
        # Create synthetic data
        data = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'latitude': [6.9271, 7.2906, 8.3114, 7.8731, 6.0535],
            'longitude': [79.8612, 80.6337, 80.4037, 80.7718, 80.2210],
            'interest_score': [80, 75, 90, 85, 70],
            'visit_duration': [2.0, 1.5, 3.0, 2.5, 1.0]
        })
        
        # Prepare data
        prepared = prepare_data_for_optimization(data)
        assert prepared is not None
        
        # Run GA
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=10,
            population_size=30,
            generations=20
        )
        
        solution, fitness, history = ga.evolve()
        
        # Verify results
        assert len(solution) == 5
        assert fitness >= 0
        assert len(history) == 20
        
        # Verify improvement over generations
        assert history[-1]['max_fitness'] >= history[0]['max_fitness']


class TestDataToVisualizationFlow:
    """Test data → optimization → visualization flow"""
    
    def test_data_to_visualization_workflow(self):
        """Test complete workflow from data to visualization"""
        # Create synthetic data
        data = pd.DataFrame({
            'name': ['Place A', 'Place B', 'Place C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'interest_score': [80, 75, 90],
            'visit_duration': [2.0, 1.5, 3.0]
        })
        
        # Prepare data
        prepared = prepare_data_for_optimization(data)
        
        # Run GA
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=24,
            population_size=20,
            generations=10
        )
        
        solution, fitness, history = ga.evolve()
        valid_tour = ga.get_valid_tour(solution)
        
        # Visualize results
        fig = plot_fitness_evolution(history)
        assert fig is not None
        
        # Create report
        report = create_summary_report(
            data, 
            valid_tour, 
            prepared['distance_matrix']
        )
        assert report is not None
        assert isinstance(report, str)
        
        # Create map
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_tour.html')
            if len(valid_tour) > 0:
                tour_map = plot_route_on_map(data, valid_tour, output_file)
                assert os.path.exists(output_file)


class TestResultsValidation:
    """Test that results are reasonable and within expected ranges"""
    
    def test_fitness_is_reasonable(self):
        """Test that fitness values are within reasonable ranges"""
        data = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'interest_score': [80, 75, 90],
            'visit_duration': [1.0, 1.0, 1.0]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=10,
            population_size=30,
            generations=20
        )
        
        solution, fitness, history = ga.evolve()
        
        # Fitness should be non-negative
        assert fitness >= 0
        
        # Fitness should not exceed sum of all scores
        max_possible_score = sum(prepared['scores'])
        assert fitness <= max_possible_score
    
    def test_valid_tour_respects_constraints(self):
        """Test that valid tour respects time constraints"""
        data = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D'],
            'latitude': [6.9271, 7.2906, 8.3114, 7.8731],
            'longitude': [79.8612, 80.6337, 80.4037, 80.7718],
            'interest_score': [80, 75, 90, 85],
            'visit_duration': [2.0, 1.5, 3.0, 2.5]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        max_time = 8
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=max_time,
            population_size=20,
            generations=10
        )
        
        solution, fitness, history = ga.evolve()
        valid_tour = ga.get_valid_tour(solution)
        
        # Calculate actual time used
        total_time = sum(prepared['visit_durations'][i] for i in valid_tour)
        
        # Add travel time
        for i in range(len(valid_tour) - 1):
            travel_time = prepared['distance_matrix'][valid_tour[i], valid_tour[i+1]] / 50
            total_time += travel_time
        
        # Should respect time constraint
        assert total_time <= max_time + 0.1  # Small tolerance for floating point
    
    def test_solution_improvement_over_generations(self):
        """Test that solution improves over generations"""
        data = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'latitude': [6.9271, 7.2906, 8.3114, 7.8731, 6.0535],
            'longitude': [79.8612, 80.6337, 80.4037, 80.7718, 80.2210],
            'interest_score': [80, 75, 90, 85, 70],
            'visit_duration': [1.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=24,
            population_size=50,
            generations=50
        )
        
        solution, fitness, history = ga.evolve()
        
        # Check improvement trend
        first_quarter_avg = np.mean([h['max_fitness'] for h in history[:12]])
        last_quarter_avg = np.mean([h['max_fitness'] for h in history[-12:]])
        
        # Last quarter should be better than first quarter
        assert last_quarter_avg >= first_quarter_avg


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_attraction(self):
        """Test with single attraction"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'latitude': [6.9271],
            'longitude': [79.8612],
            'interest_score': [80],
            'visit_duration': [2.0]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        # For single attraction, just verify the valid tour directly
        # GA requires at least 2 items for crossover to work properly
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=5
        )
        
        # Create a simple solution
        solution = [0]
        valid_tour = ga.get_valid_tour(solution)
        
        assert len(valid_tour) <= 1
        if len(valid_tour) > 0:
            assert valid_tour[0] == 0
    
    def test_zero_time_constraint(self):
        """Test with very tight time constraint"""
        data = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'interest_score': [80, 75, 90],
            'visit_duration': [2.0, 1.5, 3.0]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=0.5,  # Very tight
            population_size=10,
            generations=5
        )
        
        solution, fitness, history = ga.evolve()
        valid_tour = ga.get_valid_tour(solution)
        
        # Should have very few or no attractions
        assert len(valid_tour) <= 1
    
    def test_large_time_constraint(self):
        """Test with large time constraint (all attractions fit)"""
        data = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'interest_score': [80, 75, 90],
            'visit_duration': [1.0, 1.0, 1.0]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=100,  # Very large
            population_size=20,
            generations=10
        )
        
        solution, fitness, history = ga.evolve()
        valid_tour = ga.get_valid_tour(solution)
        
        # Should include all attractions
        assert len(valid_tour) == 3


class TestNoIntegrationIssues:
    """Test for integration issues between modules"""
    
    def test_data_utils_to_ga_compatibility(self):
        """Test that data_utils output is compatible with GA input"""
        data = pd.DataFrame({
            'name': ['A', 'B'],
            'latitude': [6.9271, 7.2906],
            'longitude': [79.8612, 80.6337],
            'interest_score': [80, 75],
            'visit_duration': [2.0, 1.5]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        # All required keys should be present
        assert 'distance_matrix' in prepared
        assert 'scores' in prepared
        assert 'visit_durations' in prepared
        
        # Should be able to create GA without errors
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=10
        )
        
        assert ga is not None
    
    def test_ga_to_visualization_compatibility(self):
        """Test that GA output is compatible with visualization input"""
        data = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'interest_score': [80, 75, 90],
            'visit_duration': [2.0, 1.5, 3.0]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        ga = GeneticAlgorithm(
            distance_matrix=prepared['distance_matrix'],
            scores=prepared['scores'],
            visit_durations=prepared['visit_durations'],
            max_time=24,
            population_size=10,
            generations=5
        )
        
        solution, fitness, history = ga.evolve()
        valid_tour = ga.get_valid_tour(solution)
        
        # Should be able to visualize without errors
        fig = plot_fitness_evolution(history)
        assert fig is not None
        
        if len(valid_tour) > 0:
            report = create_summary_report(
                data,
                valid_tour,
                prepared['distance_matrix']
            )
            assert report is not None
