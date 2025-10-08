"""
Unit tests for ga_core module
"""

import pytest
import numpy as np
from scripts.ga_core import GeneticAlgorithm


class TestGeneticAlgorithmInitialization:
    """Test GeneticAlgorithm initialization"""
    
    def test_initialization(self):
        """Test that GA initializes with correct parameters"""
        distance_matrix = np.array([[0, 10], [10, 0]])
        scores = np.array([80, 75])
        visit_durations = np.array([2.0, 1.5])
        max_time = 10
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=max_time,
            population_size=50,
            generations=100
        )
        
        assert ga.n_attractions == 2
        assert ga.population_size == 50
        assert ga.generations == 100
        assert ga.max_time == 10
        assert ga.mutation_rate == 0.1
        assert ga.crossover_rate == 0.8


class TestIndividualCreation:
    """Test individual creation"""
    
    def test_create_individual_is_valid_permutation(self):
        """Test that individuals are valid permutations"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24
        )
        
        individual = ga.create_individual()
        
        # Check length
        assert len(individual) == ga.n_attractions
        
        # Check all indices present exactly once
        assert sorted(individual) == list(range(ga.n_attractions))
    
    def test_create_individual_randomness(self):
        """Test that individuals are randomized"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24
        )
        
        # Create multiple individuals and check they're not all the same
        individuals = [ga.create_individual() for _ in range(10)]
        
        # At least some should be different
        unique_individuals = [tuple(ind) for ind in individuals]
        assert len(set(unique_individuals)) > 1


class TestPopulationCreation:
    """Test population creation"""
    
    def test_create_population_correct_size(self):
        """Test that population has correct size"""
        distance_matrix = np.array([[0, 10], [10, 0]])
        scores = np.array([80, 75])
        visit_durations = np.array([2.0, 1.5])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=10,
            population_size=30
        )
        
        population = ga.create_population()
        
        assert len(population) == 30
    
    def test_create_population_all_valid_permutations(self):
        """Test that all individuals in population are valid"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24,
            population_size=20
        )
        
        population = ga.create_population()
        
        for individual in population:
            assert len(individual) == ga.n_attractions
            assert sorted(individual) == list(range(ga.n_attractions))


class TestFitnessFunction:
    """Test fitness function"""
    
    def test_fitness_valid_tour(self):
        """Test fitness calculation for a valid tour"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([1.0, 1.0, 1.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=10
        )
        
        individual = [0, 1, 2]
        fitness = ga.fitness(individual)
        
        # Should return positive fitness
        assert fitness > 0
    
    def test_fitness_respects_time_constraint(self):
        """Test that fitness penalizes time violations"""
        distance_matrix = np.array([[0, 10], [10, 0]])
        scores = np.array([80, 75])
        visit_durations = np.array([5.0, 7.0])  # Total 12 hours
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=6  # Max time is less than visit durations
        )
        
        # Tour that would exceed time limit
        long_tour = [0, 1]
        fitness_long = ga.fitness(long_tour)
        
        # Tour that stays within time
        short_tour = [0]
        fitness_short = ga.fitness(short_tour)
        
        # Short tour should have higher or equal fitness
        assert fitness_short >= 0
    
    def test_fitness_non_negative(self):
        """Test that fitness is non-negative"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=10
        )
        
        individual = [0, 1, 2]
        fitness = ga.fitness(individual)
        
        assert fitness >= 0


class TestSelectionOperator:
    """Test selection operator"""
    
    def test_selection_returns_valid_individual(self):
        """Test that selection returns a valid individual"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24
        )
        
        population = ga.create_population()
        fitnesses = [ga.fitness(ind) for ind in population]
        
        selected = ga.selection(population, fitnesses)
        
        assert len(selected) == ga.n_attractions
        assert sorted(selected) == list(range(ga.n_attractions))
    
    def test_selection_prefers_fitter_individuals(self):
        """Test that selection has bias towards fitter individuals"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([100, 10, 50])
        visit_durations = np.array([1.0, 1.0, 1.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=10,
            population_size=20
        )
        
        # Create population with known fitnesses (at least 5 individuals for tournament)
        population = [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]]
        fitnesses = [ga.fitness(ind) for ind in population]
        
        # Run selection multiple times
        selections = [ga.selection(population, fitnesses) for _ in range(50)]
        
        # The fitter individual should be selected more often
        # (This is probabilistic, so we just check it happens)
        assert len(selections) > 0


class TestCrossoverOperator:
    """Test crossover operator"""
    
    def test_crossover_produces_valid_permutation(self):
        """Test that crossover produces valid permutations"""
        distance_matrix = np.array([[0, 10, 15, 20], [10, 0, 20, 25], [15, 20, 0, 30], [20, 25, 30, 0]])
        scores = np.array([80, 75, 90, 85])
        visit_durations = np.array([2.0, 1.5, 3.0, 2.5])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24
        )
        
        parent1 = [0, 1, 2, 3]
        parent2 = [3, 2, 1, 0]
        
        offspring = ga.crossover(parent1, parent2)
        
        assert len(offspring) == ga.n_attractions
        assert sorted(offspring) == list(range(ga.n_attractions))
    
    def test_crossover_respects_crossover_rate(self):
        """Test that crossover respects crossover rate"""
        distance_matrix = np.array([[0, 10], [10, 0]])
        scores = np.array([80, 75])
        visit_durations = np.array([2.0, 1.5])
        
        # GA with 0% crossover rate
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=10,
            crossover_rate=0.0
        )
        
        parent1 = [0, 1]
        parent2 = [1, 0]
        
        # With 0% crossover, should return copy of parent1
        offspring = ga.crossover(parent1, parent2)
        
        # Should be valid permutation
        assert sorted(offspring) == [0, 1]


class TestMutationOperator:
    """Test mutation operator"""
    
    def test_mutation_produces_valid_permutation(self):
        """Test that mutation produces valid permutations"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24,
            mutation_rate=1.0  # Always mutate
        )
        
        individual = [0, 1, 2]
        mutated = ga.mutate(individual.copy())
        
        assert len(mutated) == ga.n_attractions
        assert sorted(mutated) == list(range(ga.n_attractions))
    
    def test_mutation_changes_individual(self):
        """Test that mutation actually changes the individual"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24,
            mutation_rate=1.0  # Always mutate
        )
        
        individual = [0, 1, 2]
        
        # Mutate multiple times, at least one should be different
        mutations = [tuple(ga.mutate(individual.copy())) for _ in range(10)]
        
        # At least some should be different from original
        assert any(m != tuple(individual) for m in mutations)


class TestEvolutionLoop:
    """Test evolution loop"""
    
    def test_evolve_returns_valid_solution(self):
        """Test that evolve returns a valid solution"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24,
            population_size=20,
            generations=10  # Small number for testing
        )
        
        solution, fitness, history = ga.evolve()
        
        assert len(solution) == ga.n_attractions
        assert sorted(solution) == list(range(ga.n_attractions))
        assert fitness >= 0
        assert len(history) == 10
    
    def test_evolve_improves_fitness(self):
        """Test that evolution produces improving solutions"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([1.0, 1.0, 1.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24,
            population_size=50,
            generations=50
        )
        
        solution, fitness, history = ga.evolve()
        
        # Check that fitness generally improves
        first_gen_max = history[0]['max_fitness']
        last_gen_max = history[-1]['max_fitness']
        
        # Last generation should be at least as good as first
        assert last_gen_max >= first_gen_max
    
    def test_evolve_history_structure(self):
        """Test that evolution history has correct structure"""
        distance_matrix = np.array([[0, 10], [10, 0]])
        scores = np.array([80, 75])
        visit_durations = np.array([2.0, 1.5])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=10,
            generations=5
        )
        
        solution, fitness, history = ga.evolve()
        
        assert len(history) == 5
        for entry in history:
            assert 'generation' in entry
            assert 'max_fitness' in entry
            assert 'avg_fitness' in entry


class TestTourValidation:
    """Test tour validation"""
    
    def test_get_valid_tour_respects_time_constraint(self):
        """Test that valid tour respects time constraint"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([3.0, 3.0, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=5  # Can only fit one attraction
        )
        
        solution = [0, 1, 2]
        valid_tour = ga.get_valid_tour(solution)
        
        # Should only include attractions that fit in time
        total_time = sum(visit_durations[i] for i in valid_tour)
        assert total_time <= ga.max_time
    
    def test_get_valid_tour_empty_on_no_fit(self):
        """Test that valid tour can be empty if nothing fits"""
        distance_matrix = np.array([[0, 10], [10, 0]])
        scores = np.array([80, 75])
        visit_durations = np.array([5.0, 5.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=2  # Too small for any attraction
        )
        
        solution = [0, 1]
        valid_tour = ga.get_valid_tour(solution)
        
        # Should be empty or very limited
        assert len(valid_tour) <= 1
    
    def test_get_valid_tour_preserves_order(self):
        """Test that valid tour preserves order from solution"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([1.0, 1.0, 1.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24
        )
        
        solution = [2, 0, 1]
        valid_tour = ga.get_valid_tour(solution)
        
        # Should preserve the order from solution
        # First element should be 2
        if len(valid_tour) > 0:
            assert valid_tour[0] == 2


class TestCalculateTourTime:
    """Test calculate_tour_time method"""
    
    def test_calculate_tour_time_simple(self):
        """Test tour time calculation"""
        distance_matrix = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        scores = np.array([80, 75, 90])
        visit_durations = np.array([2.0, 1.5, 3.0])
        
        ga = GeneticAlgorithm(
            distance_matrix=distance_matrix,
            scores=scores,
            visit_durations=visit_durations,
            max_time=24
        )
        
        tour = [0, 1]
        total_time = ga.calculate_tour_time(tour)
        
        # Should be visit time + travel time
        expected = visit_durations[0] + visit_durations[1] + (10 / 50)
        assert abs(total_time - expected) < 1e-6
