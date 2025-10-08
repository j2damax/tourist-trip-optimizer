"""
Unit tests for data_utils module
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from scripts.data_utils import (
    load_attractions_data,
    calculate_distance_matrix,
    calculate_travel_time,
    validate_attraction_data,
    prepare_data_for_optimization
)


class TestLoadAttractionsData:
    """Test load_attractions_data function"""
    
    def test_load_valid_csv(self):
        """Test loading a valid CSV file"""
        # Use the actual data file from the repository
        filepath = 'data/processed/attractions.csv'
        data = load_attractions_data(filepath)
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'name' in data.columns
        assert 'latitude' in data.columns
        assert 'longitude' in data.columns
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file"""
        filepath = 'nonexistent_file.csv'
        data = load_attractions_data(filepath)
        
        assert data is None


class TestCalculateDistanceMatrix:
    """Test calculate_distance_matrix function"""
    
    def test_distance_matrix_shape(self):
        """Test that distance matrix has correct shape"""
        # Create sample data
        data = pd.DataFrame({
            'name': ['Place A', 'Place B', 'Place C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037]
        })
        
        distance_matrix = calculate_distance_matrix(data)
        
        assert distance_matrix.shape == (3, 3)
        assert isinstance(distance_matrix, np.ndarray)
    
    def test_distance_matrix_diagonal_zeros(self):
        """Test that diagonal elements are zero"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B'],
            'latitude': [6.9271, 7.2906],
            'longitude': [79.8612, 80.6337]
        })
        
        distance_matrix = calculate_distance_matrix(data)
        
        assert distance_matrix[0, 0] == 0
        assert distance_matrix[1, 1] == 0
    
    def test_distance_matrix_symmetry(self):
        """Test that distance matrix is symmetric"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B', 'Place C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037]
        })
        
        distance_matrix = calculate_distance_matrix(data)
        
        # Check symmetry
        for i in range(len(data)):
            for j in range(len(data)):
                assert abs(distance_matrix[i, j] - distance_matrix[j, i]) < 1e-10
    
    def test_distance_matrix_positive_values(self):
        """Test that distances are non-negative"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B'],
            'latitude': [6.9271, 7.2906],
            'longitude': [79.8612, 80.6337]
        })
        
        distance_matrix = calculate_distance_matrix(data)
        
        assert np.all(distance_matrix >= 0)


class TestCalculateTravelTime:
    """Test calculate_travel_time function"""
    
    def test_travel_time_default_speed(self):
        """Test travel time calculation with default speed"""
        distance = 100  # km
        travel_time = calculate_travel_time(distance)
        
        expected_time = 100 / 40  # default speed is 40 km/h
        assert travel_time == expected_time
    
    def test_travel_time_custom_speed(self):
        """Test travel time calculation with custom speed"""
        distance = 100  # km
        speed = 50  # km/h
        travel_time = calculate_travel_time(distance, avg_speed=speed)
        
        expected_time = 100 / 50
        assert travel_time == expected_time
    
    def test_travel_time_zero_distance(self):
        """Test travel time for zero distance"""
        travel_time = calculate_travel_time(0)
        assert travel_time == 0


class TestValidateAttractionData:
    """Test validate_attraction_data function"""
    
    def test_validate_complete_data_with_interest_score(self):
        """Test validation of complete data with interest_score"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'latitude': [6.9271],
            'longitude': [79.8612],
            'visit_duration': [2.0],
            'interest_score': [80]
        })
        
        assert validate_attraction_data(data) is True
    
    def test_validate_complete_data_with_score(self):
        """Test validation of complete data with score"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'latitude': [6.9271],
            'longitude': [79.8612],
            'visit_duration': [2.0],
            'score': [80]
        })
        
        assert validate_attraction_data(data) is True
    
    def test_validate_missing_name(self):
        """Test validation fails when name is missing"""
        data = pd.DataFrame({
            'latitude': [6.9271],
            'longitude': [79.8612],
            'visit_duration': [2.0],
            'interest_score': [80]
        })
        
        assert validate_attraction_data(data) is False
    
    def test_validate_missing_latitude(self):
        """Test validation fails when latitude is missing"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'longitude': [79.8612],
            'visit_duration': [2.0],
            'interest_score': [80]
        })
        
        assert validate_attraction_data(data) is False
    
    def test_validate_missing_score_columns(self):
        """Test validation fails when both score columns are missing"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'latitude': [6.9271],
            'longitude': [79.8612],
            'visit_duration': [2.0]
        })
        
        assert validate_attraction_data(data) is False


class TestPrepareDataForOptimization:
    """Test prepare_data_for_optimization function"""
    
    def test_prepare_data_success_with_interest_score(self):
        """Test successful data preparation with interest_score"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B'],
            'latitude': [6.9271, 7.2906],
            'longitude': [79.8612, 80.6337],
            'visit_duration': [2.0, 1.5],
            'interest_score': [80, 75]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        assert prepared is not None
        assert 'attractions' in prepared
        assert 'distance_matrix' in prepared
        assert 'n_attractions' in prepared
        assert 'scores' in prepared
        assert 'visit_durations' in prepared
        assert prepared['n_attractions'] == 2
        assert len(prepared['scores']) == 2
        assert len(prepared['visit_durations']) == 2
    
    def test_prepare_data_success_with_score(self):
        """Test successful data preparation with score column"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B'],
            'latitude': [6.9271, 7.2906],
            'longitude': [79.8612, 80.6337],
            'visit_duration': [2.0, 1.5],
            'score': [80, 75]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        assert prepared is not None
        assert prepared['n_attractions'] == 2
    
    def test_prepare_data_invalid_data(self):
        """Test preparation fails with invalid data"""
        data = pd.DataFrame({
            'name': ['Place A'],
            'latitude': [6.9271]
            # Missing required columns
        })
        
        prepared = prepare_data_for_optimization(data)
        
        assert prepared is None
    
    def test_prepare_data_distance_matrix_shape(self):
        """Test that prepared distance matrix has correct shape"""
        data = pd.DataFrame({
            'name': ['Place A', 'Place B', 'Place C'],
            'latitude': [6.9271, 7.2906, 8.3114],
            'longitude': [79.8612, 80.6337, 80.4037],
            'visit_duration': [2.0, 1.5, 3.0],
            'interest_score': [80, 75, 90]
        })
        
        prepared = prepare_data_for_optimization(data)
        
        assert prepared['distance_matrix'].shape == (3, 3)
