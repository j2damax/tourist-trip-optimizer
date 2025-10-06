"""
Data Preprocessing Script for Tourist Trip Optimizer

This script transforms raw location data into a clean DataFrame suitable for
the optimization algorithms, following a three-phase approach:
1. Initial Data Cleaning and Structuring
2. Data Augmentation (Synthetic Feature Generation)
3. Final Preparation for Optimization
"""

import pandas as pd
import numpy as np
from haversine import haversine, Unit
import os
import re


def categorize_location(name):
    """
    Categorize a location based on its name using keyword matching.
    
    Parameters:
    -----------
    name : str
        The location name
        
    Returns:
    --------
    str
        The category of the location
    """
    if pd.isna(name):
        return "General"
    
    name_lower = name.lower()
    
    # Religious sites
    if any(keyword in name_lower for keyword in ['temple', 'vihara', 'kovil', 'church', 'mosque', 'shrine', 'dagoba', 'stupa']):
        return "Religious Site"
    
    # Beaches
    if any(keyword in name_lower for keyword in ['beach', 'bay', 'surf']):
        return "Beach"
    
    # Nature/Wildlife
    if any(keyword in name_lower for keyword in ['national park', 'park', 'sanctuary', 'reserve', 'forest', 'wildlife']):
        return "Nature"
    
    # Waterfalls
    if any(keyword in name_lower for keyword in ['waterfall', 'falls', 'ella']):
        return "Waterfall"
    
    # Cultural/Historical
    if any(keyword in name_lower for keyword in ['fort', 'palace', 'museum', 'ruins', 'rock', 'fortress', 'statue']):
        return "Cultural"
    
    # Mountains/Hiking
    if any(keyword in name_lower for keyword in ['mountain', 'peak', 'hike', 'hiking', 'view']):
        return "Hiking"
    
    return "General"


def generate_interest_score(category):
    """
    Generate an interest score based on category with some random variation.
    
    Parameters:
    -----------
    category : str
        The location category
        
    Returns:
    --------
    float
        Interest score (0-100 scale)
    """
    base_scores = {
        "Religious Site": 80,
        "Beach": 70,
        "Nature": 90,
        "Waterfall": 75,
        "Cultural": 85,
        "Hiking": 80,
        "General": 50
    }
    
    base_score = base_scores.get(category, 50)
    # Add random variation between -5 and +5
    variation = np.random.randint(-5, 6)
    final_score = max(0, min(100, base_score + variation))
    
    return final_score


def generate_visit_duration(category):
    """
    Generate estimated visit duration based on category.
    
    Parameters:
    -----------
    category : str
        The location category
        
    Returns:
    --------
    float
        Visit duration in hours
    """
    durations = {
        "Religious Site": 1.5,
        "Beach": 2.0,
        "Nature": 3.0,
        "Waterfall": 1.5,
        "Cultural": 2.0,
        "Hiking": 3.0,
        "General": 1.0
    }
    
    return durations.get(category, 1.0)


def phase1_clean_data(raw_data_path):
    """
    Phase 1: Initial Data Cleaning and Structuring
    
    - Identify and select core columns (name, latitude, longitude)
    - Handle missing values
    - Convert data types and validate coordinates
    - Remove duplicates
    
    Parameters:
    -----------
    raw_data_path : str
        Path to the raw CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with name, latitude, longitude columns
    """
    print("=" * 80)
    print("PHASE 1: Initial Data Cleaning and Structuring")
    print("=" * 80)
    
    # Load raw data
    print(f"\nLoading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Filter out non-tourist locations (mechanics, fuel stations, restaurants, accommodation)
    print("\n0. Filtering tourist attractions only...")
    if 'category' in df.columns:
        initial_rows = len(df)
        # Exclude non-tourist categories
        exclude_categories = ['Mechanics', 'FuelStations', 'Fuel Stations', 'Restaurants', 'Accommodation']
        df = df[~df['category'].isin(exclude_categories)]
        filtered_rows = initial_rows - len(df)
        print(f"   Filtered out {filtered_rows} non-tourist locations")
        print(f"   Remaining rows: {len(df)}")
    
    # Select and rename core columns
    print("\n1. Selecting core columns...")
    # The raw data has: location_id, name, latitude, longitude, category, rating
    core_columns = {
        'name': 'name',
        'latitude': 'latitude',
        'longitude': 'longitude'
    }
    
    # Check if all required columns exist
    missing_cols = [col for col in core_columns.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create new DataFrame with selected columns
    df_clean = df[list(core_columns.keys())].copy()
    print(f"   Selected {len(core_columns)} core columns")
    
    # Handle missing values
    print("\n2. Handling missing values...")
    print(f"   Missing values before:")
    print(f"      name: {df_clean['name'].isna().sum()}")
    print(f"      latitude: {df_clean['latitude'].isna().sum()}")
    print(f"      longitude: {df_clean['longitude'].isna().sum()}")
    
    # Drop rows with missing essential data
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['name', 'latitude', 'longitude'])
    rows_dropped = initial_rows - len(df_clean)
    print(f"   Dropped {rows_dropped} rows with missing essential data")
    
    # Data type conversion and validation
    print("\n3. Converting data types and validating coordinates...")
    df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
    df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
    
    # Drop any rows where conversion failed
    df_clean = df_clean.dropna(subset=['latitude', 'longitude'])
    
    # Validate Sri Lanka coordinate ranges (Latitude: 5-10°, Longitude: 79-82°)
    initial_rows = len(df_clean)
    df_clean = df_clean[
        (df_clean['latitude'] >= 5) & (df_clean['latitude'] <= 10) &
        (df_clean['longitude'] >= 79) & (df_clean['longitude'] <= 82)
    ]
    invalid_coords = initial_rows - len(df_clean)
    print(f"   Removed {invalid_coords} rows with invalid coordinates")
    print(f"   Latitude range: [{df_clean['latitude'].min():.2f}, {df_clean['latitude'].max():.2f}]")
    print(f"   Longitude range: [{df_clean['longitude'].min():.2f}, {df_clean['longitude'].max():.2f}]")
    
    # Deduplication
    print("\n4. Removing duplicates...")
    initial_rows = len(df_clean)
    
    # Remove duplicate names (case-insensitive)
    df_clean = df_clean.drop_duplicates(subset=['name'], keep='first')
    name_dups = initial_rows - len(df_clean)
    
    # Remove duplicates based on exact coordinates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
    coord_dups = initial_rows - len(df_clean)
    
    print(f"   Removed {name_dups} duplicate names")
    print(f"   Removed {coord_dups} duplicate coordinates")
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"\nPhase 1 Complete. Final shape: {df_clean.shape}")
    
    return df_clean


def phase2_augment_data(df):
    """
    Phase 2: Data Augmentation (Synthetic Feature Generation)
    
    - Create category column based on location names
    - Generate interest_score based on category
    - Generate visit_duration based on category
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame from Phase 1
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional columns: category, interest_score, visit_duration
    """
    print("\n" + "=" * 80)
    print("PHASE 2: Data Augmentation (Synthetic Feature Generation)")
    print("=" * 80)
    
    df_augmented = df.copy()
    
    # Create category column
    print("\n1. Creating category column based on name patterns...")
    df_augmented['category'] = df_augmented['name'].apply(categorize_location)
    
    category_counts = df_augmented['category'].value_counts()
    print("   Category distribution:")
    for cat, count in category_counts.items():
        print(f"      {cat}: {count}")
    
    # Generate interest_score
    print("\n2. Generating interest scores...")
    np.random.seed(42)  # For reproducibility
    df_augmented['interest_score'] = df_augmented['category'].apply(generate_interest_score)
    
    print(f"   Interest score range: [{df_augmented['interest_score'].min():.1f}, {df_augmented['interest_score'].max():.1f}]")
    print(f"   Mean interest score: {df_augmented['interest_score'].mean():.1f}")
    
    # Generate visit_duration
    print("\n3. Generating visit durations...")
    df_augmented['visit_duration'] = df_augmented['category'].apply(generate_visit_duration)
    
    print(f"   Visit duration range: [{df_augmented['visit_duration'].min():.1f}, {df_augmented['visit_duration'].max():.1f}] hours")
    print(f"   Mean visit duration: {df_augmented['visit_duration'].mean():.1f} hours")
    
    # Add poi_id column (index-based)
    df_augmented['poi_id'] = range(len(df_augmented))
    
    # Reorder columns for better readability
    column_order = ['poi_id', 'name', 'latitude', 'longitude', 'category', 'interest_score', 'visit_duration']
    df_augmented = df_augmented[column_order]
    
    print(f"\nPhase 2 Complete. Final shape: {df_augmented.shape}")
    print(f"Columns: {list(df_augmented.columns)}")
    
    return df_augmented


def phase3_prepare_optimization(df, avg_speed_kmh=40):
    """
    Phase 3: Final Preparation for Optimization
    
    - Calculate distance matrix using Haversine formula
    - Convert to travel time matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        Augmented DataFrame from Phase 2
    avg_speed_kmh : float
        Average travel speed in km/h (default: 40)
        
    Returns:
    --------
    tuple
        (DataFrame, distance_matrix, travel_time_matrix)
    """
    print("\n" + "=" * 80)
    print("PHASE 3: Final Preparation for Optimization")
    print("=" * 80)
    
    n = len(df)
    
    # Calculate distance matrix
    print(f"\n1. Calculating distance matrix for {n} locations...")
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                loc1 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
                loc2 = (df.iloc[j]['latitude'], df.iloc[j]['longitude'])
                distance_matrix[i, j] = haversine(loc1, loc2, unit=Unit.KILOMETERS)
    
    print(f"   Distance matrix shape: {distance_matrix.shape}")
    print(f"   Distance range: [{distance_matrix[distance_matrix > 0].min():.2f}, {distance_matrix.max():.2f}] km")
    print(f"   Mean distance: {distance_matrix[distance_matrix > 0].mean():.2f} km")
    
    # Convert to travel time matrix
    print(f"\n2. Converting to travel time matrix (avg speed: {avg_speed_kmh} km/h)...")
    travel_time_matrix = distance_matrix / avg_speed_kmh
    
    print(f"   Travel time matrix shape: {travel_time_matrix.shape}")
    print(f"   Travel time range: [{travel_time_matrix[travel_time_matrix > 0].min():.2f}, {travel_time_matrix.max():.2f}] hours")
    print(f"   Mean travel time: {travel_time_matrix[travel_time_matrix > 0].mean():.2f} hours")
    
    print(f"\nPhase 3 Complete.")
    
    return df, distance_matrix, travel_time_matrix


def preprocess_and_save(raw_data_path, output_dir, avg_speed_kmh=40):
    """
    Complete preprocessing pipeline: clean, augment, and prepare data for optimization.
    
    Parameters:
    -----------
    raw_data_path : str
        Path to raw CSV file
    output_dir : str
        Directory to save processed files
    avg_speed_kmh : float
        Average travel speed in km/h (default: 40)
    """
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING PIPELINE FOR TOURIST TRIP OPTIMIZER")
    print("=" * 80)
    
    # Phase 1: Clean data
    df_clean = phase1_clean_data(raw_data_path)
    
    # Phase 2: Augment data
    df_augmented = phase2_augment_data(df_clean)
    
    # Phase 3: Prepare for optimization
    df_final, distance_matrix, travel_time_matrix = phase3_prepare_optimization(df_augmented, avg_speed_kmh)
    
    # Save outputs
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed attractions data
    attractions_file = os.path.join(output_dir, 'attractions.csv')
    df_final.to_csv(attractions_file, index=False)
    print(f"\n✓ Saved attractions data to: {attractions_file}")
    print(f"  Rows: {len(df_final)}, Columns: {len(df_final.columns)}")
    
    # Save distance matrix
    distance_matrix_file = os.path.join(output_dir, 'distance_matrix.npy')
    np.save(distance_matrix_file, distance_matrix)
    print(f"\n✓ Saved distance matrix to: {distance_matrix_file}")
    print(f"  Shape: {distance_matrix.shape}")
    
    # Save travel time matrix
    travel_time_matrix_file = os.path.join(output_dir, 'travel_time_matrix.npy')
    np.save(travel_time_matrix_file, travel_time_matrix)
    print(f"\n✓ Saved travel time matrix to: {travel_time_matrix_file}")
    print(f"  Shape: {travel_time_matrix.shape}")
    
    # Create a summary report
    summary_file = os.path.join(output_dir, 'preprocessing_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("DATA PREPROCESSING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input file: {raw_data_path}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Processing date: {pd.Timestamp.now()}\n\n")
        f.write(f"Final dataset:\n")
        f.write(f"  - Number of attractions: {len(df_final)}\n")
        f.write(f"  - Columns: {', '.join(df_final.columns)}\n\n")
        f.write(f"Category distribution:\n")
        for cat, count in df_final['category'].value_counts().items():
            f.write(f"  - {cat}: {count}\n")
        f.write(f"\nInterest scores:\n")
        f.write(f"  - Range: [{df_final['interest_score'].min():.1f}, {df_final['interest_score'].max():.1f}]\n")
        f.write(f"  - Mean: {df_final['interest_score'].mean():.1f}\n")
        f.write(f"  - Std: {df_final['interest_score'].std():.1f}\n")
        f.write(f"\nVisit durations (hours):\n")
        f.write(f"  - Range: [{df_final['visit_duration'].min():.1f}, {df_final['visit_duration'].max():.1f}]\n")
        f.write(f"  - Mean: {df_final['visit_duration'].mean():.1f}\n")
        f.write(f"  - Total: {df_final['visit_duration'].sum():.1f}\n")
        f.write(f"\nDistance matrix:\n")
        f.write(f"  - Shape: {distance_matrix.shape}\n")
        f.write(f"  - Range: [{distance_matrix[distance_matrix > 0].min():.2f}, {distance_matrix.max():.2f}] km\n")
        f.write(f"  - Mean: {distance_matrix[distance_matrix > 0].mean():.2f} km\n")
        f.write(f"\nTravel time matrix (avg speed: {avg_speed_kmh} km/h):\n")
        f.write(f"  - Shape: {travel_time_matrix.shape}\n")
        f.write(f"  - Range: [{travel_time_matrix[travel_time_matrix > 0].min():.2f}, {travel_time_matrix.max():.2f}] hours\n")
        f.write(f"  - Mean: {travel_time_matrix[travel_time_matrix > 0].mean():.2f} hours\n")
    
    print(f"\n✓ Saved summary report to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nProcessed {len(df_final)} attractions ready for optimization.")
    print(f"Files saved to: {output_dir}")
    
    return df_final, distance_matrix, travel_time_matrix


if __name__ == "__main__":
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'locations.csv')
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    # Run preprocessing pipeline
    df, distance_matrix, travel_time_matrix = preprocess_and_save(
        raw_data_path=raw_data_path,
        output_dir=output_dir,
        avg_speed_kmh=40  # As specified in the problem statement
    )
    
    # Display sample of processed data
    print("\n" + "=" * 80)
    print("SAMPLE OF PROCESSED DATA (first 10 rows)")
    print("=" * 80)
    print(df.head(10).to_string(index=False))
