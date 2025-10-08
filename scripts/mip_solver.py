"""
MIP Solver for Tourist Trip Design Problem

This module implements a Mixed Integer Programming (MIP) solver using PuLP
to solve the Tourist Trip Design Problem (TTDP) as a benchmark approach.
The solver formulates the problem as an optimization model and finds the
optimal tour that maximizes tourist satisfaction while respecting time constraints.
"""

import numpy as np
import pandas as pd
from pulp import *
import time
from typing import Tuple, List, Dict, Optional


class MIPSolver:
    """
    Mixed Integer Programming solver for the Tourist Trip Design Problem.
    
    Uses PuLP library to formulate and solve the TTDP as a MIP optimization
    problem with objective to maximize total satisfaction score subject to
    time constraints and tour structure requirements.
    """
    
    def __init__(self, distance_matrix: np.ndarray, scores: np.ndarray, 
                 visit_durations: np.ndarray, max_time: float, 
                 avg_speed: float = 50.0):
        """
        Initialize the MIP Solver.
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            Matrix of distances between attractions (in kilometers)
        scores : np.ndarray
            Satisfaction scores for each attraction
        visit_durations : np.ndarray
            Time required to visit each attraction (in hours)
        max_time : float
            Maximum time available for the trip (in hours)
        avg_speed : float, optional
            Average travel speed in km/h (default: 50.0)
        """
        self.distance_matrix = distance_matrix
        self.scores = scores
        self.visit_durations = visit_durations
        self.max_time = max_time
        self.avg_speed = avg_speed
        self.n_attractions = len(scores)
        
        # Model components (initialized during build)
        self.model = None
        self.x = None  # Visit variables
        self.y = None  # Travel variables
        self.u = None  # Position variables (MTZ)
        
        # Solution (populated after solve)
        self.solution_status = None
        self.computation_time = None
        self.objective_value = None
        self.visited_attractions = None
        self.tour_sequence = None
        
    def build_model(self):
        """
        Build the MIP model with decision variables, objective, and constraints.
        
        This method creates:
        - Decision variables (visit, travel, position)
        - Objective function (maximize total score)
        - Time constraint
        - Flow conservation constraints
        - Subtour elimination constraints (MTZ formulation)
        """
        print("Building MIP model...")
        start_time = time.time()
        
        # Create the optimization model
        self.model = LpProblem("Tourist_Trip_Design", LpMaximize)
        
        # Decision variables
        self._create_decision_variables()
        
        # Objective function
        self._add_objective_function()
        
        # Constraints
        self._add_time_constraint()
        self._add_flow_conservation_constraints()
        self._add_subtour_elimination_constraints()
        
        build_time = time.time() - start_time
        print(f"Model built in {build_time:.2f} seconds")
        print(f"Number of variables: {len(self.model.variables())}")
        print(f"Number of constraints: {len(self.model.constraints)}")
        
    def _create_decision_variables(self):
        """
        Create decision variables for the MIP model.
        
        Variables:
        - x[i]: Binary variable, 1 if attraction i is visited, 0 otherwise
        - y[i,j]: Binary variable, 1 if we travel from attraction i to j, 0 otherwise
        - u[i]: Continuous variable for position in tour (for subtour elimination)
        """
        # x[i] = 1 if attraction i is visited
        self.x = LpVariable.dicts("visit", range(self.n_attractions), cat='Binary')
        
        # y[i,j] = 1 if we travel from attraction i to j
        self.y = LpVariable.dicts("travel",
                                  [(i, j) for i in range(self.n_attractions) 
                                   for j in range(self.n_attractions) if i != j],
                                  cat='Binary')
        
        # u[i] = position of attraction i in the tour (for MTZ subtour elimination)
        self.u = LpVariable.dicts("position", range(self.n_attractions),
                                  lowBound=0, upBound=self.n_attractions-1, 
                                  cat='Continuous')
        
    def _add_objective_function(self):
        """
        Add objective function: Maximize total satisfaction score.
        
        Objective: max Σ(scores[i] * x[i]) for all attractions i
        """
        self.model += lpSum([self.scores[i] * self.x[i] 
                            for i in range(self.n_attractions)]), "Total_Score"
        
    def _add_time_constraint(self):
        """
        Add time constraint: Total time (visit + travel) ≤ max_time.
        
        Total time = Σ(visit_durations[i] * x[i]) + Σ(distance[i,j] / speed * y[i,j])
        """
        # Total visit time at attractions
        total_visit_time = lpSum([self.visit_durations[i] * self.x[i] 
                                 for i in range(self.n_attractions)])
        
        # Total travel time between attractions
        total_travel_time = lpSum([self.distance_matrix[i][j] / self.avg_speed * self.y[(i, j)]
                                  for i in range(self.n_attractions) 
                                  for j in range(self.n_attractions) if i != j])
        
        # Add constraint
        self.model += total_visit_time + total_travel_time <= self.max_time, "Time_Limit"
        
    def _add_flow_conservation_constraints(self):
        """
        Add flow conservation constraints for tour structure.
        
        For each visited attraction:
        - Number of outgoing edges = 1 if visited, 0 otherwise
        - Number of incoming edges = 1 if visited, 0 otherwise
        
        This ensures that each visited attraction is entered once and left once.
        """
        for i in range(self.n_attractions):
            # Outgoing flow: sum of y[i,j] for all j ≠ i equals x[i]
            self.model += lpSum([self.y[(i, j)] for j in range(self.n_attractions) 
                               if i != j]) == self.x[i], f"Outflow_{i}"
            
            # Incoming flow: sum of y[j,i] for all j ≠ i equals x[i]
            self.model += lpSum([self.y[(j, i)] for j in range(self.n_attractions) 
                               if i != j]) == self.x[i], f"Inflow_{i}"
            
    def _add_subtour_elimination_constraints(self):
        """
        Add Miller-Tucker-Zemlin (MTZ) subtour elimination constraints.
        
        MTZ constraints: u[i] - u[j] + n * y[i,j] ≤ n - 1 for all i ≠ j
        
        These constraints ensure that the solution forms a single connected tour
        without disconnected subtours.
        """
        n = self.n_attractions
        
        for i in range(n):
            for j in range(n):
                if i != j and i > 0 and j > 0:  # Exclude depot (attraction 0)
                    self.model += (self.u[i] - self.u[j] + n * self.y[(i, j)] 
                                  <= n - 1), f"MTZ_{i}_{j}"
                    
    def solve(self, time_limit: int = 300, verbose: bool = True):
        """
        Solve the MIP model using the CBC solver.
        
        Parameters:
        -----------
        time_limit : int, optional
            Maximum time in seconds for solver (default: 300)
        verbose : bool, optional
            Whether to print solver messages (default: True)
            
        Returns:
        --------
        dict
            Dictionary containing solution results with keys:
            - 'status': Solution status (e.g., 'Optimal', 'Feasible')
            - 'objective_value': Total satisfaction score achieved
            - 'computation_time': Time taken to solve (seconds)
            - 'tour_sequence': List of attraction indices in visit order
            - 'visited_attractions': List of visited attraction indices
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
            
        print("Solving MIP model...")
        start_solve = time.time()
        
        # Configure and run solver
        solver = PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit)
        self.model.solve(solver)
        
        # Record computation time
        self.computation_time = time.time() - start_solve
        
        # Extract solution
        self.solution_status = LpStatus[self.model.status]
        self.objective_value = value(self.model.objective) if self.model.status == 1 else None
        
        print(f"\nSolution status: {self.solution_status}")
        print(f"Computation time: {self.computation_time:.2f} seconds")
        
        if self.objective_value is not None:
            print(f"Objective value (Total Score): {self.objective_value:.2f}")
            
        # Extract tour from solution
        self._extract_solution()
        
        return self.get_solution_dict()
        
    def _extract_solution(self):
        """
        Extract the tour sequence and visited attractions from the solved model.
        
        Builds the tour by following the travel variables (y[i,j]) that are set to 1.
        """
        # Extract visited attractions
        self.visited_attractions = [i for i in range(self.n_attractions) 
                                   if value(self.x[i]) > 0.5]
        
        if not self.visited_attractions:
            self.tour_sequence = []
            return
            
        # Extract tour edges
        tour_edges = [(i, j) for i in range(self.n_attractions) 
                     for j in range(self.n_attractions)
                     if i != j and value(self.y[(i, j)]) > 0.5]
        
        # Reconstruct tour sequence from edges
        if tour_edges:
            # Create adjacency dictionary
            next_attraction = {i: j for i, j in tour_edges}
            
            # Start from first attraction in the edges
            self.tour_sequence = []
            current = list(next_attraction.keys())[0]
            self.tour_sequence.append(current)
            
            visited_set = {current}
            while current in next_attraction and next_attraction[current] not in visited_set:
                current = next_attraction[current]
                self.tour_sequence.append(current)
                visited_set.add(current)
        else:
            # No edges, return visited attractions as sequence
            self.tour_sequence = self.visited_attractions
            
    def get_solution_dict(self) -> Dict:
        """
        Get solution results as a dictionary.
        
        Returns:
        --------
        dict
            Dictionary with solution information including status, objective value,
            computation time, and tour details.
        """
        return {
            'status': self.solution_status,
            'objective_value': self.objective_value,
            'computation_time': self.computation_time,
            'tour_sequence': self.tour_sequence,
            'visited_attractions': self.visited_attractions,
            'n_attractions_visited': len(self.visited_attractions) if self.visited_attractions else 0
        }
        
    def verify_solution(self) -> Dict:
        """
        Verify that the solution satisfies all constraints and compute actual metrics.
        
        Returns:
        --------
        dict
            Dictionary with verification results including:
            - 'total_visit_time': Time spent at attractions (hours)
            - 'total_travel_time': Time spent traveling (hours)
            - 'total_travel_distance': Total distance traveled (km)
            - 'total_time': Total time used (hours)
            - 'total_score': Total satisfaction score achieved
            - 'time_constraint_satisfied': Whether time constraint is satisfied
        """
        if not self.tour_sequence:
            return {
                'total_visit_time': 0,
                'total_travel_time': 0,
                'total_travel_distance': 0,
                'total_time': 0,
                'total_score': 0,
                'time_constraint_satisfied': True
            }
            
        # Calculate total visit time
        total_visit_time = sum([self.visit_durations[i] for i in self.tour_sequence])
        
        # Calculate total travel time and distance
        total_travel_time = 0
        total_travel_dist = 0
        
        for i in range(len(self.tour_sequence) - 1):
            dist = self.distance_matrix[self.tour_sequence[i], self.tour_sequence[i+1]]
            total_travel_dist += dist
            total_travel_time += dist / self.avg_speed
            
        total_time = total_visit_time + total_travel_time
        total_score = sum([self.scores[i] for i in self.tour_sequence])
        
        verification = {
            'total_visit_time': total_visit_time,
            'total_travel_time': total_travel_time,
            'total_travel_distance': total_travel_dist,
            'total_time': total_time,
            'total_score': total_score,
            'time_constraint_satisfied': total_time <= self.max_time
        }
        
        return verification
        
    def print_solution(self, attractions_df: Optional[pd.DataFrame] = None):
        """
        Print the solution in a human-readable format.
        
        Parameters:
        -----------
        attractions_df : pd.DataFrame, optional
            DataFrame with attraction information (must have 'name' column)
            If provided, prints attraction names instead of indices.
        """
        if self.solution_status is None:
            print("No solution available. Run solve() first.")
            return
            
        print("\n" + "="*60)
        print("MIP SOLUTION SUMMARY")
        print("="*60)
        
        print(f"\nStatus: {self.solution_status}")
        print(f"Computation time: {self.computation_time:.2f} seconds")
        
        if self.objective_value is not None:
            print(f"Objective value: {self.objective_value:.2f}")
            print(f"Number of attractions visited: {len(self.visited_attractions)}")
            
            if attractions_df is not None and 'name' in attractions_df.columns:
                print("\nVisited attractions:")
                for idx in self.visited_attractions:
                    print(f"  - {attractions_df.iloc[idx]['name']} (Score: {self.scores[idx]})")
                    
                if self.tour_sequence:
                    print("\nTour sequence:")
                    for i, idx in enumerate(self.tour_sequence, 1):
                        print(f"  {i}. {attractions_df.iloc[idx]['name']}")
            else:
                print(f"\nVisited attraction indices: {self.visited_attractions}")
                print(f"Tour sequence: {self.tour_sequence}")
                
            # Print verification
            verification = self.verify_solution()
            print("\n" + "-"*60)
            print("SOLUTION VERIFICATION")
            print("-"*60)
            print(f"Total visit time: {verification['total_visit_time']:.2f} hours")
            print(f"Total travel distance: {verification['total_travel_distance']:.2f} km")
            print(f"Total travel time: {verification['total_travel_time']:.2f} hours")
            print(f"Total time: {verification['total_time']:.2f} hours (limit: {self.max_time} hours)")
            print(f"Total score: {verification['total_score']:.2f}")
            print(f"Time constraint satisfied: {verification['time_constraint_satisfied']}")
            
        print("="*60 + "\n")


def solve_ttdp_mip(distance_matrix: np.ndarray, scores: np.ndarray,
                   visit_durations: np.ndarray, max_time: float,
                   avg_speed: float = 50.0, time_limit: int = 300,
                   verbose: bool = True) -> Dict:
    """
    Convenience function to solve TTDP using MIP in one call.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Matrix of distances between attractions (in kilometers)
    scores : np.ndarray
        Satisfaction scores for each attraction
    visit_durations : np.ndarray
        Time required to visit each attraction (in hours)
    max_time : float
        Maximum time available for the trip (in hours)
    avg_speed : float, optional
        Average travel speed in km/h (default: 50.0)
    time_limit : int, optional
        Maximum solver time in seconds (default: 300)
    verbose : bool, optional
        Whether to print progress messages (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing solution results
        
    Example:
    --------
    >>> import numpy as np
    >>> from mip_solver import solve_ttdp_mip
    >>> 
    >>> # Load your data
    >>> distance_matrix = np.load('distance_matrix.npy')
    >>> scores = np.array([9.5, 9.0, 8.5, ...])
    >>> visit_durations = np.array([3.0, 2.5, 2.0, ...])
    >>> 
    >>> # Solve
    >>> result = solve_ttdp_mip(distance_matrix, scores, visit_durations, 
    ...                         max_time=24, avg_speed=50)
    >>> 
    >>> print(f"Best tour: {result['tour_sequence']}")
    >>> print(f"Total score: {result['objective_value']}")
    """
    solver = MIPSolver(distance_matrix, scores, visit_durations, max_time, avg_speed)
    solver.build_model()
    solution = solver.solve(time_limit=time_limit, verbose=verbose)
    
    return solution
