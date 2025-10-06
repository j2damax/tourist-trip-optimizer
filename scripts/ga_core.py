"""
Genetic Algorithm Core Implementation for Tourist Trip Design Problem

This module implements a genetic algorithm to optimize tourist trip routes
by maximizing tourist satisfaction while considering time constraints.
"""

import numpy as np
import random
from typing import List, Tuple


class GeneticAlgorithm:
    """
    Genetic Algorithm for solving the Tourist Trip Design Problem (TTDP).
    """
    
    def __init__(self, distance_matrix, scores, visit_durations, 
                 max_time, population_size=100, generations=500,
                 mutation_rate=0.1, crossover_rate=0.8):
        """
        Initialize the Genetic Algorithm.
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            Matrix of distances between attractions
        scores : np.ndarray
            Satisfaction scores for each attraction
        visit_durations : np.ndarray
            Time required to visit each attraction (in hours)
        max_time : float
            Maximum time available for the trip (in hours)
        population_size : int
            Size of the population
        generations : int
            Number of generations to run
        mutation_rate : float
            Probability of mutation
        crossover_rate : float
            Probability of crossover
        """
        self.distance_matrix = distance_matrix
        self.scores = scores
        self.visit_durations = visit_durations
        self.max_time = max_time
        self.n_attractions = len(scores)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def create_individual(self):
        """
        Create a random individual (tour).
        
        Returns:
        --------
        list
            Random permutation of attraction indices
        """
        individual = list(range(self.n_attractions))
        random.shuffle(individual)
        return individual
    
    def create_population(self):
        """
        Create initial population.
        
        Returns:
        --------
        list
            List of individuals
        """
        return [self.create_individual() for _ in range(self.population_size)]
    
    def calculate_tour_time(self, tour):
        """
        Calculate total time required for a tour.
        
        Parameters:
        -----------
        tour : list
            List of attraction indices representing a tour
            
        Returns:
        --------
        float
            Total time required (travel + visit time)
        """
        total_time = 0
        
        for i in range(len(tour)):
            total_time += self.visit_durations[tour[i]]
            if i < len(tour) - 1:
                travel_time = self.distance_matrix[tour[i], tour[i+1]] / 50  # Assume 50 km/h
                total_time += travel_time
        
        return total_time
    
    def fitness(self, individual):
        """
        Calculate fitness of an individual.
        
        Parameters:
        -----------
        individual : list
            Tour represented as list of attraction indices
            
        Returns:
        --------
        float
            Fitness score (total satisfaction if within time constraint, penalized otherwise)
        """
        total_time = 0
        total_score = 0
        
        for i, attraction_idx in enumerate(individual):
            total_time += self.visit_durations[attraction_idx]
            
            if total_time > self.max_time:
                break
            
            total_score += self.scores[attraction_idx]
            
            if i < len(individual) - 1:
                travel_time = self.distance_matrix[attraction_idx, individual[i+1]] / 50
                total_time += travel_time
                
                if total_time > self.max_time:
                    break
        
        if total_time > self.max_time:
            penalty = (total_time - self.max_time) * 10
            return max(0, total_score - penalty)
        
        return total_score
    
    def selection(self, population, fitnesses):
        """
        Tournament selection.
        
        Parameters:
        -----------
        population : list
            Current population
        fitnesses : list
            Fitness values for each individual
            
        Returns:
        --------
        list
            Selected individual
        """
        tournament_size = 5
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """
        Order crossover (OX) operator.
        
        Parameters:
        -----------
        parent1 : list
            First parent
        parent2 : list
            Second parent
            
        Returns:
        --------
        list
            Offspring
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        offspring = [-1] * size
        offspring[start:end] = parent1[start:end]
        
        pointer = end
        for gene in parent2[end:] + parent2[:end]:
            if gene not in offspring:
                if pointer >= size:
                    pointer = 0
                offspring[pointer] = gene
                pointer += 1
        
        return offspring
    
    def mutate(self, individual):
        """
        Swap mutation operator.
        
        Parameters:
        -----------
        individual : list
            Individual to mutate
            
        Returns:
        --------
        list
            Mutated individual
        """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        
        return individual
    
    def evolve(self):
        """
        Run the genetic algorithm.
        
        Returns:
        --------
        tuple
            (best_solution, best_fitness, fitness_history)
        """
        population = self.create_population()
        fitness_history = []
        best_solution = None
        best_fitness = -np.inf
        
        for generation in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            
            max_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            fitness_history.append({'generation': generation, 
                                   'max_fitness': max_fitness,
                                   'avg_fitness': avg_fitness})
            
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_solution = population[fitnesses.index(max_fitness)].copy()
            
            new_population = []
            
            for _ in range(self.population_size):
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population.append(offspring)
            
            population = new_population
        
        return best_solution, best_fitness, fitness_history
    
    def get_valid_tour(self, solution):
        """
        Extract valid tour within time constraint.
        
        Parameters:
        -----------
        solution : list
            Full tour solution
            
        Returns:
        --------
        list
            Valid tour within time constraint
        """
        valid_tour = []
        total_time = 0
        
        for attraction_idx in solution:
            total_time += self.visit_durations[attraction_idx]
            
            if total_time > self.max_time:
                break
            
            valid_tour.append(attraction_idx)
            
            if len(valid_tour) < len(solution):
                travel_time = self.distance_matrix[attraction_idx, solution[len(valid_tour)]] / 50
                if total_time + travel_time > self.max_time:
                    break
                total_time += travel_time
        
        return valid_tour
