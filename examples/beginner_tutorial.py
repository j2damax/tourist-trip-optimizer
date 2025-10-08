"""
Beginner Tutorial: Your First Genetic Algorithm
================================================

This script demonstrates how to solve a simple tourist trip optimization
problem using a Genetic Algorithm. Perfect for learning!

Time to complete: 10-15 minutes
Prerequisites: Basic Python knowledge
"""

import sys
sys.path.append('../scripts')

import pandas as pd
import numpy as np
from data_utils import calculate_distance_matrix
from ga_core import GeneticAlgorithm

def main():
    print("=" * 70)
    print("BEGINNER TUTORIAL: Genetic Algorithm for Trip Optimization")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Create Sample Data
    # ========================================================================
    print("\n📍 STEP 1: Creating sample attractions...")
    
    # Simple dataset with 5 attractions
    attractions = pd.DataFrame({
        'name': ['Ancient Temple', 'Tropical Beach', 'Mountain Fort', 
                 'Wildlife Safari', 'Art Museum'],
        'latitude': [7.29, 6.03, 6.93, 6.37, 6.50],
        'longitude': [80.64, 80.22, 79.85, 81.52, 80.10],
        'score': [9.5, 7.0, 8.8, 9.2, 6.5],  # Satisfaction rating (0-10)
        'visit_duration': [2.0, 1.5, 2.5, 4.0, 1.0]  # Hours to spend
    })
    
    print(f"✓ Created {len(attractions)} attractions")
    print(f"\nAttractions:")
    for idx, row in attractions.iterrows():
        print(f"  {idx}. {row['name']:20s} - Score: {row['score']}, Duration: {row['visit_duration']}h")
    
    # ========================================================================
    # STEP 2: Calculate Distances
    # ========================================================================
    print("\n📏 STEP 2: Calculating distances between attractions...")
    
    distance_matrix = calculate_distance_matrix(attractions)
    
    print(f"✓ Distance matrix calculated ({len(attractions)}x{len(attractions)})")
    print(f"  Example: {attractions.iloc[0]['name']} to {attractions.iloc[1]['name']}: " +
          f"{distance_matrix[0,1]:.1f} km")
    
    # ========================================================================
    # STEP 3: Set Up the Problem
    # ========================================================================
    print("\n⚙️  STEP 3: Configuring the optimization problem...")
    
    MAX_TIME = 12  # You have 12 hours available
    
    print(f"✓ Time available: {MAX_TIME} hours")
    print(f"  Total possible visit time: {attractions['visit_duration'].sum():.1f} hours")
    print(f"  → Can't visit everything! Need to optimize.")
    
    # ========================================================================
    # STEP 4: Initialize Genetic Algorithm
    # ========================================================================
    print("\n🧬 STEP 4: Initializing Genetic Algorithm...")
    
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        scores=attractions['score'].values,
        visit_durations=attractions['visit_duration'].values,
        max_time=MAX_TIME,
        population_size=50,      # 50 random tours in each generation
        generations=100,          # Evolve for 100 generations
        mutation_rate=0.1,        # 10% chance of random change
        crossover_rate=0.8        # 80% chance of combining parents
    )
    
    print(f"✓ GA configured with:")
    print(f"  - Population size: 50 tours")
    print(f"  - Generations: 100")
    print(f"  - Mutation rate: 10%")
    print(f"  - Crossover rate: 80%")
    
    # ========================================================================
    # STEP 5: Run Optimization
    # ========================================================================
    print("\n🚀 STEP 5: Running optimization...")
    print("  (This may take a few seconds...)")
    
    best_solution, best_fitness, history = ga.evolve()
    
    print(f"✓ Optimization complete!")
    
    # ========================================================================
    # STEP 6: Analyze Results
    # ========================================================================
    print("\n📊 STEP 6: Results Analysis")
    print("=" * 70)
    
    # Get feasible tour (within time limit)
    tour = ga.get_valid_tour(best_solution)
    
    # Calculate tour statistics
    total_visit_time = sum(attractions.iloc[tour]['visit_duration'])
    total_score = sum(attractions.iloc[tour]['score'])
    
    # Calculate travel time
    total_travel_time = 0
    for i in range(len(tour) - 1):
        distance = distance_matrix[tour[i], tour[i+1]]
        total_travel_time += distance / 50  # Assume 50 km/h
    
    total_time = total_visit_time + total_travel_time
    
    # Display results
    print(f"\n🏆 BEST TOUR FOUND:")
    print("-" * 70)
    for i, idx in enumerate(tour):
        print(f"  {i+1}. {attractions.iloc[idx]['name']:20s} " +
              f"(Score: {attractions.iloc[idx]['score']}, " +
              f"Duration: {attractions.iloc[idx]['visit_duration']}h)")
        if i < len(tour) - 1:
            distance = distance_matrix[tour[i], tour[i+1]]
            print(f"      ↓ Travel {distance:.1f} km ({distance/50:.1f}h)")
    
    print("-" * 70)
    print(f"\n📈 STATISTICS:")
    print(f"  Attractions visited: {len(tour)} out of {len(attractions)}")
    print(f"  Total satisfaction score: {total_score:.1f}")
    print(f"  Total visit time: {total_visit_time:.1f} hours")
    print(f"  Total travel time: {total_travel_time:.1f} hours")
    print(f"  Total time used: {total_time:.1f} / {MAX_TIME} hours")
    print(f"  Time remaining: {MAX_TIME - total_time:.1f} hours")
    
    # Show evolution progress
    print(f"\n📉 EVOLUTION PROGRESS:")
    print(f"  Generation 0 (random):    Best = {history[0]['max_fitness']:.2f}")
    print(f"  Generation 50:            Best = {history[50]['max_fitness']:.2f}")
    print(f"  Generation 100 (final):   Best = {history[-1]['max_fitness']:.2f}")
    print(f"  Improvement: {history[-1]['max_fitness'] - history[0]['max_fitness']:.2f} points " +
          f"({((history[-1]['max_fitness']/history[0]['max_fitness'])-1)*100:.1f}%)")
    
    # ========================================================================
    # STEP 7: What's Next?
    # ========================================================================
    print("\n" + "=" * 70)
    print("🎓 WHAT YOU LEARNED:")
    print("=" * 70)
    print("✓ How to set up an optimization problem")
    print("✓ How to run a Genetic Algorithm")
    print("✓ How to interpret the results")
    print("✓ How evolution improves solutions over time")
    
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS:")
    print("=" * 70)
    print("1. Read LEARNING_GUIDE.md for deeper understanding")
    print("2. Run notebooks/05_Interactive_Learning_Tutorial.ipynb")
    print("3. Try modifying parameters above (population_size, generations, etc.)")
    print("4. Add your own attractions to the dataset")
    print("5. Run notebooks/02_Genetic_Algorithm_Implementation.ipynb with full data")
    
    print("\n" + "=" * 70)
    print("💡 EXPERIMENTS TO TRY:")
    print("=" * 70)
    print("- Change MAX_TIME to 8 or 16 hours - what happens?")
    print("- Increase generations to 200 - does it improve?")
    print("- Set mutation_rate to 0.3 - how does it affect convergence?")
    print("- Add a 6th attraction - how does the solution change?")
    
    print("\n🎉 Congratulations! You've run your first optimization algorithm!")
    print("=" * 70)

if __name__ == "__main__":
    main()
