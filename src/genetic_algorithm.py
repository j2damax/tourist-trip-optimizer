"""
Genetic Algorithm implementation for Tourist Trip Design Problem (TTDP).

This module implements a GA to find near-optimal multi-day tourist itineraries.
"""

import numpy as np
import random
from typing import List, Tuple


class GeneticAlgorithmTTDP:
    """
    Genetic Algorithm solver for the Tourist Trip Design Problem.
    
    Uses permutation-based chromosome representation where each chromosome
    is a sequence of POI IDs that gets decoded into a multi-day itinerary.
    """
    
    def __init__(self, pois_df, travel_time_matrix, num_days=3, 
                 max_time_per_day=8, population_size=200, 
                 generations=500, crossover_rate=0.85, mutation_rate=0.03,
                 tournament_size=3):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            pois_df: DataFrame with POI data (must include 'interest_score' and 'visit_duration')
            travel_time_matrix: NxN matrix of travel times between POIs
            num_days: Number of days for the trip
            max_time_per_day: Maximum hours available per day
            population_size: Size of the population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
        """
        self.pois_df = pois_df
        self.travel_time_matrix = travel_time_matrix
        self.num_days = num_days
        self.max_time_per_day = max_time_per_day
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        self.num_pois = len(pois_df)
        self.poi_indices = list(range(self.num_pois))
        
        # For tracking convergence
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def decode_chromosome(self, chromosome):
        """
        Decode a chromosome (permutation of POI IDs) into a multi-day itinerary.
        
        Args:
            chromosome: List of POI indices in order
            
        Returns:
            Tuple of (itinerary, total_score) where itinerary is a list of lists
        """
        itinerary = []
        current_day = []
        current_time = 0
        total_score = 0
        
        for poi_idx in chromosome:
            visit_time = self.pois_df.iloc[poi_idx]['visit_duration']
            
            # Calculate travel time from previous POI or start of day
            if len(current_day) == 0:
                travel_time = 0  # Starting the day
            else:
                prev_poi = current_day[-1]
                travel_time = self.travel_time_matrix[prev_poi, poi_idx]
            
            # Check if adding this POI exceeds daily time budget
            new_time = current_time + travel_time + visit_time
            
            if new_time <= self.max_time_per_day:
                current_day.append(poi_idx)
                current_time = new_time
                total_score += self.pois_df.iloc[poi_idx]['interest_score']
            else:
                # Day is full, start new day if available
                if len(itinerary) < self.num_days:
                    itinerary.append(current_day)
                    current_day = [poi_idx]
                    current_time = visit_time
                    total_score += self.pois_df.iloc[poi_idx]['interest_score']
                else:
                    # No more days available, stop
                    break
        
        # Add the last day if it has POIs
        if current_day and len(itinerary) < self.num_days:
            itinerary.append(current_day)
        
        return itinerary, total_score
    
    def fitness(self, chromosome):
        """Calculate fitness as total interest score of decoded itinerary."""
        _, total_score = self.decode_chromosome(chromosome)
        return total_score
    
    def initialize_population(self):
        """Create initial population of random permutations."""
        population = []
        for _ in range(self.population_size):
            chromosome = self.poi_indices.copy()
            random.shuffle(chromosome)
            population.append(chromosome)
        return population
    
    def tournament_selection(self, population, fitness_scores):
        """Select a parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def ordered_crossover(self, parent1, parent2):
        """
        Perform Ordered Crossover (OX1) on two parent chromosomes.
        
        Preserves relative ordering of POIs from parents.
        """
        size = len(parent1)
        
        # Select two random crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Create offspring
        offspring1 = [None] * size
        offspring2 = [None] * size
        
        # Copy segment between crossover points
        offspring1[point1:point2] = parent1[point1:point2]
        offspring2[point1:point2] = parent2[point1:point2]
        
        # Fill remaining positions from other parent
        def fill_offspring(offspring, donor):
            pos = point2
            for gene in donor[point2:] + donor[:point2]:
                if gene not in offspring:
                    if pos >= size:
                        pos = 0
                    offspring[pos] = gene
                    pos += 1
                    if pos == point1:
                        pos = point2
        
        fill_offspring(offspring1, parent2)
        fill_offspring(offspring2, parent1)
        
        return offspring1, offspring2
    
    def swap_mutation(self, chromosome):
        """Perform swap mutation on a chromosome."""
        mutated = chromosome.copy()
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    def evolve(self, verbose=True):
        """
        Run the genetic algorithm.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Tuple of (best_chromosome, best_itinerary, best_score)
        """
        # Initialize population
        population = self.initialize_population()
        
        best_overall_chromosome = None
        best_overall_fitness = -1
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.fitness(chromo) for chromo in population]
            
            # Track best
            max_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.best_fitness_history.append(max_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            if max_fitness > best_overall_fitness:
                best_overall_fitness = max_fitness
                best_overall_chromosome = population[np.argmax(fitness_scores)].copy()
            
            if verbose and (generation % 50 == 0 or generation == self.generations - 1):
                print(f"Generation {generation}: Best={max_fitness:.1f}, Avg={avg_fitness:.1f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best chromosome
            new_population.append(best_overall_chromosome.copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self.ordered_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = self.swap_mutation(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self.swap_mutation(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            population = new_population
        
        # Decode best solution
        best_itinerary, best_score = self.decode_chromosome(best_overall_chromosome)
        
        return best_overall_chromosome, best_itinerary, best_score
