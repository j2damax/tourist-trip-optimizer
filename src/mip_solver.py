"""
Mixed-Integer Programming (MIP) solver for Tourist Trip Design Problem (TTDP).

This module implements an exact MIP formulation to find optimal itineraries.
"""

import numpy as np
import pandas as pd
from pulp import *


class MIPSolverTTDP:
    """
    MIP solver for the Tourist Trip Design Problem.
    
    Formulates and solves the TTDP as a Mixed-Integer Linear Program
    to find the mathematically optimal solution.
    """
    
    def __init__(self, pois_df, travel_time_matrix, num_days=3, 
                 max_time_per_day=8, time_limit=300):
        """
        Initialize the MIP solver.
        
        Args:
            pois_df: DataFrame with POI data (must include 'interest_score' and 'visit_duration')
            travel_time_matrix: NxN matrix of travel times between POIs
            num_days: Number of days for the trip
            max_time_per_day: Maximum hours available per day
            time_limit: Maximum solver time in seconds
        """
        self.pois_df = pois_df
        self.travel_time_matrix = travel_time_matrix
        self.num_days = num_days
        self.max_time_per_day = max_time_per_day
        self.time_limit = time_limit
        
        self.num_pois = len(pois_df)
        self.poi_indices = list(range(self.num_pois))
        self.days = list(range(num_days))
        
        # We use index 0 as the hotel (dummy POI)
        # All actual POIs are indexed 1 to num_pois
        self.V = [0] + list(range(1, self.num_pois + 1))
        self.V_without_hotel = list(range(1, self.num_pois + 1))
        
    def solve(self, verbose=True):
        """
        Solve the TTDP using MIP formulation.
        
        Returns:
            Tuple of (itinerary, total_score, status)
        """
        # Create the problem
        prob = LpProblem("TTDP", LpMaximize)
        
        # Decision variables
        # y[i,d] = 1 if POI i is visited on day d
        y = LpVariable.dicts("y", 
                            ((i, d) for i in self.V_without_hotel for d in self.days),
                            cat='Binary')
        
        # x[i,j,d] = 1 if we travel from POI i to POI j on day d
        x = LpVariable.dicts("x",
                            ((i, j, d) for i in self.V for j in self.V for d in self.days),
                            cat='Binary')
        
        # u[i,d] = position of POI i in the sequence on day d (for subtour elimination)
        u = LpVariable.dicts("u",
                            ((i, d) for i in self.V_without_hotel for d in self.days),
                            lowBound=0,
                            upBound=self.num_pois,
                            cat='Continuous')
        
        # Objective: Maximize total interest score
        prob += lpSum([self.pois_df.iloc[i-1]['interest_score'] * y[i,d] 
                      for i in self.V_without_hotel for d in self.days])
        
        # Constraint 1: Each POI visited at most once during the entire trip
        for i in self.V_without_hotel:
            prob += lpSum([y[i,d] for d in self.days]) <= 1
        
        # Constraint 2: Flow conservation - if we visit a POI, we arrive and depart
        for i in self.V_without_hotel:
            for d in self.days:
                prob += lpSum([x[j,i,d] for j in self.V if j != i]) == y[i,d]
                prob += lpSum([x[i,j,d] for j in self.V if j != i]) == y[i,d]
        
        # Constraint 3: Each day starts and ends at hotel (POI 0)
        for d in self.days:
            prob += lpSum([x[0,j,d] for j in self.V_without_hotel]) <= 1
            prob += lpSum([x[i,0,d] for i in self.V_without_hotel]) <= 1
            prob += lpSum([x[0,j,d] for j in self.V_without_hotel]) == lpSum([x[i,0,d] for i in self.V_without_hotel])
        
        # Constraint 4: Daily time budget
        for d in self.days:
            # Travel time - need to map V indices to actual matrix indices
            travel_time = lpSum([
                self.travel_time_matrix[
                    0 if i == 0 else i-1,  # Map hotel (0) and POIs (1-N) to matrix (0 to N-1)
                    0 if j == 0 else j-1
                ] * x[i,j,d] 
                for i in self.V for j in self.V if i != j
            ])
            # Visit time
            visit_time = lpSum([self.pois_df.iloc[i-1]['visit_duration'] * y[i,d] 
                              for i in self.V_without_hotel])
            prob += travel_time + visit_time <= self.max_time_per_day
        
        # Constraint 5: Subtour elimination (Miller-Tucker-Zemlin formulation)
        for i in self.V_without_hotel:
            for j in self.V_without_hotel:
                if i != j:
                    for d in self.days:
                        prob += u[i,d] - u[j,d] + (self.num_pois + 1) * x[i,j,d] <= self.num_pois
        
        # No self-loops
        for i in self.V:
            for d in self.days:
                prob += x[i,i,d] == 0
        
        # Solve the problem
        if verbose:
            print("Starting MIP solver...")
        
        solver = PULP_CBC_CMD(timeLimit=self.time_limit, msg=verbose)
        prob.solve(solver)
        
        status = LpStatus[prob.status]
        
        if verbose:
            print(f"Status: {status}")
            print(f"Objective Value: {value(prob.objective):.1f}")
        
        # Extract solution
        itinerary = []
        total_score = 0
        
        if status == "Optimal" or status == "Feasible":
            for d in self.days:
                day_route = []
                
                # Find POIs visited on this day
                visited_pois = [i for i in self.V_without_hotel if value(y[i,d]) > 0.5]
                
                if visited_pois:
                    # Build the route by following x variables
                    current = 0  # Start at hotel
                    route = [0]
                    
                    while True:
                        # Find next POI
                        next_poi = None
                        for j in self.V:
                            if j != current and value(x[current,j,d]) > 0.5:
                                next_poi = j
                                break
                        
                        if next_poi is None or next_poi == 0:
                            break
                        
                        route.append(next_poi)
                        current = next_poi
                    
                    # Convert to actual POI indices (subtract 1 for DataFrame indexing)
                    day_route = [poi - 1 for poi in route if poi > 0]
                    
                    # Calculate score for this day
                    for poi_idx in day_route:
                        total_score += self.pois_df.iloc[poi_idx]['interest_score']
                
                if day_route:
                    itinerary.append(day_route)
        
        return itinerary, total_score, status
