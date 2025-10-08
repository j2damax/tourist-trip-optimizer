"""
Unit tests for visualization module
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from scripts.visualization import (
    _get_score_column,
    plot_fitness_evolution,
    plot_route_on_map,
    plot_distance_matrix,
    plot_tour_statistics,
    compare_algorithms,
    create_summary_report
)


class TestGetScoreColumn:
    """Test _get_score_column helper function"""
    
    def test_get_score_column_interest_score(self):
        """Test that it prefers interest_score"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'interest_score': [80],
            'score': [70]
        })
        
        score_col = _get_score_column(data)
        assert score_col == 'interest_score'
    
    def test_get_score_column_score(self):
        """Test that it falls back to score"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'score': [70]
        })
        
        score_col = _get_score_column(data)
        assert score_col == 'score'
    
    def test_get_score_column_missing(self):
        """Test that it raises error when no score column exists"""
        data = pd.DataFrame({
            'name': ['Place A']
        })
        
        with pytest.raises(ValueError):
            _get_score_column(data)


class TestPlotFitnessEvolution:
    """Test plot_fitness_evolution function"""
    
    def test_plot_fitness_evolution_executes(self):
        """Test that plot function executes without errors"""
        fitness_history = [
            {'generation': 0, 'max_fitness': 100, 'avg_fitness': 80},
            {'generation': 1, 'max_fitness': 120, 'avg_fitness': 90},
            {'generation': 2, 'max_fitness': 130, 'avg_fitness': 95}
        ]
        
        fig = plot_fitness_evolution(fitness_history)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_fitness_evolution_returns_figure(self):
        """Test that it returns a matplotlib figure"""
        fitness_history = [
            {'generation': 0, 'max_fitness': 100, 'avg_fitness': 80},
            {'generation': 1, 'max_fitness': 120, 'avg_fitness': 90}
        ]
        
        fig = plot_fitness_evolution(fitness_history)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotRouteOnMap:
    """Test plot_route_on_map function"""
    
    def test_plot_route_on_map_executes(self):
        """Test that map plotting executes without errors"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B', 'Place C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'interest_score': [80, 75, 90],
            'visit_duration': [2.0, 1.5, 3.0]
        })
        
        tour_indices = [0, 1, 2]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_map.html')
            tour_map = plot_route_on_map(data, tour_indices, output_file)
            
            assert tour_map is not None
            assert os.path.exists(output_file)
    
    def test_plot_route_on_map_empty_tour(self):
        """Test handling of empty tour"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'latitude': [6.9271],
            'longitude': [79.8612],
            'interest_score': [80],
            'visit_duration': [2.0]
        })
        
        tour_indices = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_map.html')
            tour_map = plot_route_on_map(data, tour_indices, output_file)
            
            # Should return None for empty tour
            assert tour_map is None


class TestPlotDistanceMatrix:
    """Test plot_distance_matrix function"""
    
    def test_plot_distance_matrix_executes(self):
        """Test that distance matrix plotting executes"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        
        fig = plot_distance_matrix(distance_matrix)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_distance_matrix_with_names(self):
        """Test plotting with attraction names"""
        distance_matrix = np.array([[0, 10], [10, 0]])
        attraction_names = ['Place A', 'Place B']
        
        fig = plot_distance_matrix(distance_matrix, attraction_names)
        
        assert fig is not None
        plt.close(fig)


class TestPlotTourStatistics:
    """Test plot_tour_statistics function"""
    
    def test_plot_tour_statistics_executes(self):
        """Test that tour statistics plotting executes"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B', 'Place C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'interest_score': [80, 75, 90],
            'visit_duration': [2.0, 1.5, 3.0]
        })
        
        tour_indices = [0, 1, 2]
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        
        fig = plot_tour_statistics(data, tour_indices, distance_matrix)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_tour_statistics_single_attraction(self):
        """Test plotting with single attraction"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'latitude': [6.9271],
            'longitude': [79.8612],
            'interest_score': [80],
            'visit_duration': [2.0]
        })
        
        tour_indices = [0]
        distance_matrix = np.array([[0]])
        
        fig = plot_tour_statistics(data, tour_indices, distance_matrix)
        
        assert fig is not None
        plt.close(fig)


class TestCompareAlgorithms:
    """Test compare_algorithms function"""
    
    def test_compare_algorithms_executes(self):
        """Test that algorithm comparison executes"""
        results_dict = {
            'GA': {'score': 245, 'computation_time': 5.2},
            'Random': {'score': 180, 'computation_time': 0.1}
        }
        
        fig = compare_algorithms(results_dict)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_compare_algorithms_multiple(self):
        """Test comparison with multiple algorithms"""
        results_dict = {
            'GA': {'score': 245, 'computation_time': 5.2},
            'Random': {'score': 180, 'computation_time': 0.1},
            'Greedy': {'score': 220, 'computation_time': 0.5}
        }
        
        fig = compare_algorithms(results_dict)
        
        assert fig is not None
        plt.close(fig)


class TestCreateSummaryReport:
    """Test create_summary_report function"""
    
    def test_create_summary_report_executes(self):
        """Test that summary report creation executes"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B', 'Place C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'interest_score': [80, 75, 90],
            'visit_duration': [2.0, 1.5, 3.0]
        })
        
        tour_indices = [0, 1, 2]
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        
        report = create_summary_report(data, tour_indices, distance_matrix)
        
        assert report is not None
        assert isinstance(report, str)
    
    def test_create_summary_report_contains_key_info(self):
        """Test that report contains key information"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B'],
            'latitude': [6.9271, 7.2906],
            'longitude': [79.8612, 80.6337],
            'interest_score': [80, 75],
            'visit_duration': [2.0, 1.5]
        })
        
        tour_indices = [0, 1]
        distance_matrix = np.array([[0, 10], [10, 0]])
        
        report = create_summary_report(data, tour_indices, distance_matrix, algorithm_name='TestGA')
        
        # Check that report contains important information
        assert 'Place A' in report
        assert 'Place B' in report
        assert 'TestGA' in report
        assert 'Total Attractions' in report
        assert 'Total Satisfaction Score' in report
    
    def test_create_summary_report_with_score_column(self):
        """Test report creation with 'score' column instead of 'interest_score'"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B'],
            'latitude': [6.9271, 7.2906],
            'longitude': [79.8612, 80.6337],
            'score': [80, 75],
            'visit_duration': [2.0, 1.5]
        })
        
        tour_indices = [0, 1]
        distance_matrix = np.array([[0, 10], [10, 0]])
        
        report = create_summary_report(data, tour_indices, distance_matrix)
        
        assert report is not None
        assert isinstance(report, str)
        assert 'Place A' in report
