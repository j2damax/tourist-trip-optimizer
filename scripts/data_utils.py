"""
Data Utilities for Tourist Trip Design Problem

This module provides utility functions for loading, processing, and 
managing tourist attraction data for Sri Lankan tourist destinations.
"""

import pandas as pd
import numpy as np
from haversine import haversine, Unit


def load_attractions_data(filepath):
    """
    Load tourist attractions data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing attraction data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing attraction information
    """
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None


def calculate_distance_matrix(attractions):
    """
    Calculate distance matrix between all attractions using haversine formula.
    
    Parameters:
    -----------
    attractions : pd.DataFrame
        DataFrame containing attraction data with 'latitude' and 'longitude' columns
        
    Returns:
    --------
    np.ndarray
        Distance matrix (in kilometers)
    """
    n = len(attractions)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                loc1 = (attractions.iloc[i]['latitude'], attractions.iloc[i]['longitude'])
                loc2 = (attractions.iloc[j]['latitude'], attractions.iloc[j]['longitude'])
                distance_matrix[i, j] = haversine(loc1, loc2, unit=Unit.KILOMETERS)
    
    return distance_matrix


def calculate_travel_time(distance, avg_speed=50):
    """
    Calculate travel time based on distance and average speed.
    
    Parameters:
    -----------
    distance : float
        Distance in kilometers
    avg_speed : float
        Average speed in km/h (default: 50)
        
    Returns:
    --------
    float
        Travel time in hours
    """
    return distance / avg_speed


def validate_attraction_data(data):
    """
    Validate that the attraction data contains required columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing attraction data
        
    Returns:
    --------
    bool
        True if data is valid, False otherwise
    """
    required_columns = ['name', 'latitude', 'longitude', 'visit_duration', 'score']
    
    for col in required_columns:
        if col not in data.columns:
            print(f"Error: Missing required column '{col}'")
            return False
    
    return True


def prepare_data_for_optimization(data):
    """
    Prepare attraction data for optimization algorithms.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw attraction data
        
    Returns:
    --------
    dict
        Dictionary containing prepared data for optimization
    """
    if not validate_attraction_data(data):
        return None
    
    distance_matrix = calculate_distance_matrix(data)
    
    prepared_data = {
        'attractions': data,
        'distance_matrix': distance_matrix,
        'n_attractions': len(data),
        'scores': data['score'].values,
        'visit_durations': data['visit_duration'].values
    }
    
    return prepared_data
