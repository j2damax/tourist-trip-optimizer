"""
Visualization Utilities for Tourist Trip Design Problem

This module provides functions for visualizing tourist trip routes,
optimization results, and performance metrics.
"""

import matplotlib.pyplot as plt
import folium
import numpy as np
import pandas as pd


def plot_fitness_evolution(fitness_history):
    """
    Plot the evolution of fitness over generations.
    
    Parameters:
    -----------
    fitness_history : list
        List of dictionaries containing fitness statistics per generation
    """
    generations = [h['generation'] for h in fitness_history]
    max_fitness = [h['max_fitness'] for h in fitness_history]
    avg_fitness = [h['avg_fitness'] for h in fitness_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fitness, label='Best Fitness', linewidth=2)
    plt.plot(generations, avg_fitness, label='Average Fitness', linewidth=2, alpha=0.7)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.title('Genetic Algorithm Fitness Evolution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_route_on_map(attractions_data, tour_indices, output_file='tour_map.html'):
    """
    Plot the tourist route on an interactive map using Folium.
    
    Parameters:
    -----------
    attractions_data : pd.DataFrame
        DataFrame containing attraction data with 'name', 'latitude', 'longitude'
    tour_indices : list
        List of attraction indices representing the tour
    output_file : str
        Path to save the HTML map file
        
    Returns:
    --------
    folium.Map
        Interactive map object
    """
    if len(tour_indices) == 0:
        print("Warning: Empty tour provided")
        return None
    
    center_lat = attractions_data.iloc[tour_indices]['latitude'].mean()
    center_lon = attractions_data.iloc[tour_indices]['longitude'].mean()
    
    tour_map = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    for idx, attraction_idx in enumerate(tour_indices):
        attraction = attractions_data.iloc[attraction_idx]
        
        folium.Marker(
            location=[attraction['latitude'], attraction['longitude']],
            popup=f"{idx+1}. {attraction['name']}<br>Score: {attraction['score']}<br>Duration: {attraction['visit_duration']}h",
            tooltip=attraction['name'],
            icon=folium.Icon(color='red' if idx == 0 else 'blue' if idx == len(tour_indices)-1 else 'green',
                           icon='info-sign')
        ).add_to(tour_map)
    
    route_coordinates = [
        [attractions_data.iloc[idx]['latitude'], attractions_data.iloc[idx]['longitude']]
        for idx in tour_indices
    ]
    
    folium.PolyLine(
        route_coordinates,
        color='blue',
        weight=3,
        opacity=0.7,
        popup='Tour Route'
    ).add_to(tour_map)
    
    tour_map.save(output_file)
    print(f"Map saved to {output_file}")
    
    return tour_map


def plot_distance_matrix(distance_matrix, attraction_names=None):
    """
    Visualize the distance matrix as a heatmap.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Matrix of distances between attractions
    attraction_names : list, optional
        Names of attractions for axis labels
    """
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Distance (km)')
    
    if attraction_names is not None:
        plt.xticks(range(len(attraction_names)), attraction_names, rotation=90)
        plt.yticks(range(len(attraction_names)), attraction_names)
    
    plt.xlabel('Attraction', fontsize=12)
    plt.ylabel('Attraction', fontsize=12)
    plt.title('Distance Matrix Heatmap', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()


def plot_tour_statistics(attractions_data, tour_indices, distance_matrix):
    """
    Create a comprehensive visualization of tour statistics.
    
    Parameters:
    -----------
    attractions_data : pd.DataFrame
        DataFrame containing attraction data
    tour_indices : list
        List of attraction indices in the tour
    distance_matrix : np.ndarray
        Matrix of distances between attractions
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    tour_attractions = attractions_data.iloc[tour_indices]
    
    axes[0, 0].bar(range(len(tour_indices)), tour_attractions['score'].values)
    axes[0, 0].set_xlabel('Stop Number')
    axes[0, 0].set_ylabel('Satisfaction Score')
    axes[0, 0].set_title('Satisfaction Score by Stop')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(range(len(tour_indices)), tour_attractions['visit_duration'].values, color='orange')
    axes[0, 1].set_xlabel('Stop Number')
    axes[0, 1].set_ylabel('Visit Duration (hours)')
    axes[0, 1].set_title('Visit Duration by Stop')
    axes[0, 1].grid(True, alpha=0.3)
    
    distances = []
    for i in range(len(tour_indices) - 1):
        dist = distance_matrix[tour_indices[i], tour_indices[i+1]]
        distances.append(dist)
    
    if distances:
        axes[1, 0].plot(range(len(distances)), distances, marker='o', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('Distance (km)')
        axes[1, 0].set_title('Distance Between Consecutive Stops')
        axes[1, 0].grid(True, alpha=0.3)
    
    total_score = tour_attractions['score'].sum()
    total_visit_time = tour_attractions['visit_duration'].sum()
    total_travel_dist = sum(distances) if distances else 0
    total_travel_time = total_travel_dist / 50  # Assume 50 km/h
    
    stats_text = f"""
    Total Attractions: {len(tour_indices)}
    Total Score: {total_score:.2f}
    Total Visit Time: {total_visit_time:.2f}h
    Total Travel Distance: {total_travel_dist:.2f}km
    Total Travel Time: {total_travel_time:.2f}h
    Total Time: {total_visit_time + total_travel_time:.2f}h
    """
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].set_title('Tour Summary Statistics')
    
    plt.tight_layout()
    
    return fig


def compare_algorithms(results_dict):
    """
    Compare performance of different algorithms.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with algorithm names as keys and results as values
    """
    algorithms = list(results_dict.keys())
    scores = [results_dict[alg]['score'] for alg in algorithms]
    times = [results_dict[alg]['computation_time'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(algorithms, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel('Total Satisfaction Score', fontsize=12)
    ax1.set_title('Solution Quality Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(algorithms, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Computation Time Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig


def create_summary_report(attractions_data, tour_indices, distance_matrix, algorithm_name='GA'):
    """
    Create a comprehensive summary report of the tour.
    
    Parameters:
    -----------
    attractions_data : pd.DataFrame
        DataFrame containing attraction data
    tour_indices : list
        List of attraction indices in the tour
    distance_matrix : np.ndarray
        Matrix of distances between attractions
    algorithm_name : str
        Name of the algorithm used
        
    Returns:
    --------
    str
        Formatted summary report
    """
    tour_attractions = attractions_data.iloc[tour_indices]
    
    total_score = tour_attractions['score'].sum()
    total_visit_time = tour_attractions['visit_duration'].sum()
    
    distances = []
    for i in range(len(tour_indices) - 1):
        dist = distance_matrix[tour_indices[i], tour_indices[i+1]]
        distances.append(dist)
    
    total_travel_dist = sum(distances) if distances else 0
    total_travel_time = total_travel_dist / 50
    total_time = total_visit_time + total_travel_time
    
    report = f"""
{'='*60}
TOURIST TRIP OPTIMIZATION REPORT
{'='*60}

Algorithm: {algorithm_name}

Tour Sequence:
{'-'*60}
"""
    
    for idx, attraction_idx in enumerate(tour_indices):
        attraction = attractions_data.iloc[attraction_idx]
        report += f"{idx+1}. {attraction['name']}\n"
        report += f"   Score: {attraction['score']:.2f} | Duration: {attraction['visit_duration']:.2f}h\n"
        
        if idx < len(tour_indices) - 1:
            dist = distances[idx]
            travel_time = dist / 50
            report += f"   → Travel: {dist:.2f}km ({travel_time:.2f}h)\n"
        report += "\n"
    
    report += f"""
{'-'*60}
Summary Statistics:
{'-'*60}
Total Attractions Visited: {len(tour_indices)}
Total Satisfaction Score: {total_score:.2f}
Total Visit Time: {total_visit_time:.2f} hours
Total Travel Distance: {total_travel_dist:.2f} km
Total Travel Time: {total_travel_time:.2f} hours
Total Trip Duration: {total_time:.2f} hours

{'='*60}
"""
    
    return report
