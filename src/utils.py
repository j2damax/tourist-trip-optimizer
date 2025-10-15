"""
Utility functions for the Sri Lanka Itinerary Optimization project.

This module contains reusable helper functions for data processing,
distance calculations, and other common operations.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two geographic coordinates using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point
        lat2, lon2: Latitude and longitude of the second point
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r


def load_pois(filepath):
    """
    Load Points of Interest from a CSV file.
    
    Args:
        filepath: Path to the POIs CSV file
    
    Returns:
        DataFrame containing POI data
    """
    return pd.read_csv(filepath)


def calculate_travel_time(distance_km, avg_speed_kmh=40):
    """
    Calculate travel time in hours based on distance and average speed.
    
    Args:
        distance_km: Distance in kilometers
        avg_speed_kmh: Average speed in km/h (default 40 km/h for Sri Lanka)
    
    Returns:
        Travel time in hours
    """
    return distance_km / avg_speed_kmh


def assign_interest_score(category, rating):
    """
    Assign interest score based on POI category and rating.
    Simulates user preferences for history and nature.
    
    Args:
        category: POI category
        rating: POI rating (1-5)
    
    Returns:
        Interest score (1-100)
    """
    # Base score ranges by category
    category_ranges = {
        'Cultural': (70, 100),  # HistoricalSites equivalent
        'Wildlife': (70, 100),  # Nature Reserves equivalent
        'HikesnViews': (70, 100),  # Nature related
        'UniqueExperiences': (40, 70),
        'Beaches': (30, 60),
        'Waterfalls': (50, 80),
        'Accommodation': (10, 30),
        'default': (10, 40)
    }
    
    # Get the range for the category
    min_score, max_score = category_ranges.get(category, category_ranges['default'])
    
    # Scale based on rating (1-5)
    if rating >= 1 and rating <= 5:
        # Linear interpolation based on rating
        score = min_score + (max_score - min_score) * ((rating - 1) / 4)
    else:
        # If no rating, use midpoint
        score = (min_score + max_score) / 2
    
    return int(score)


def assign_visit_duration(category, rating):
    """
    Assign visit duration in hours based on POI category and rating.
    
    Args:
        category: POI category
        rating: POI rating (1-5)
    
    Returns:
        Visit duration in hours
    """
    # Major sites get longer durations
    major_categories = ['Cultural', 'Wildlife', 'UniqueExperiences', 'HikesnViews']
    standard_categories = ['Beaches', 'Waterfalls']
    
    if category in major_categories and rating >= 4:
        return 3.0  # Major sites
    elif category in major_categories or category in standard_categories:
        return 1.5  # Standard attractions
    else:
        return 0.5  # Minor viewpoints
